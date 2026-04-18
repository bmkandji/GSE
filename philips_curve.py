from gse_engine.ahlgrim.tools import *
from scipy.optimize import minimize
from .ornstein_ulhenbeck import Ornstein_Uhlenbeck
from scipy import stats
from typing import Optional
from gse_engine.db import read_sql_sheet


class Phillips_curve:
    def __init__(self, infla_model: Ornstein_Uhlenbeck, params=None):
        """
        Initialise le modèle OU et estime par maximum de vraisemblance les paramètres
        d'un modèle AR(1) appliqué aux log-variations extraites des données.

        Les paramètres estimés sont :
            - α (alpha) : constante
            - φ (phi) : coefficient AR(1)
            - σ (sigma) : écart-type de l'erreur
            - κ = -log(φ) (kappa)
            - μ = α/(1-φ) (mu)
            - la log-vraisemblance (log_lik)
            - le R² de la régression

        :type infla_model: object
        """
        self._type = "PC"
        self.df_path = {"path": None, "index": "Index"}
        self.inflate = infla_model
        self.df = None
        self.df_brut = None
        self.ajustement_var = None
        self.year_col = None
        self.value_col = None
        self.type_data = None
        self.start_year = None
        self.end_year = None
        if params is None:
            params = [None] * 5
        # Paramètres estimés
        self.kappa, self.mu, self.alpha, self.sigma, self.rho = params
        self.sigma_epsilon = None
        self.residuals = None  # Stocke les résidus
        self.sw_test = None
        self.correlation_epsilon_mq = None
        self.log_lik = None  # log-vraisemblance
        self.r_squared = None
        self.r_pseudo_squared = None
        self.z0 = None  # Condition initiale pour q_t

        # Séries de log-variations
        self.m_t = None
        self.m_tm1 = None
        self.q_tm1 = None
        self.x_t = None
        self.x_tm1 = None

    def _prepare_data(self):
        """Tronquer les données à partir de start_year et préparer les séries pour la régression."""
        try:
            # Charger les métadonnées
            dico = read_sql_sheet(self.df_path["index"], self.df_path["path"])

            # Vérifier si les colonnes nécessaires sont présentes
            if "Refrence" not in dico.columns or "data" not in dico.columns:
                raise ValueError(
                    "Le fichier Index doit contenir les colonnes 'Refrence' et 'data'."
                )

            # Extraire les noms des colonnes pour les dates et valeurs
            self.year_col = dico.loc[dico["Refrence"] == "Date", "data"].values[0]
            self.value_col = dico.loc[dico["Refrence"] == "Variable", "data"].values[0]
            self.start_year = int(
                dico.loc[dico["Refrence"] == "start_year", "data"].values[0]
            )
            self.end_year = int(
                dico.loc[dico["Refrence"] == "end_year", "data"].values[0]
            )
            self.type_data = dico.loc[dico["Refrence"] == "Type", "data"].values[0]
            self.df_path["data"] = dico.loc[
                dico["Refrence"] == "DataLeaf", "data"
            ].values[0]

            if len(self.year_col) == 0 or len(self.value_col) == 0:
                raise ValueError(
                    "Les colonnes pour 'Date' et 'Variable' n'ont pas été trouvées dans le fichier Index."
                )

            # Charger les données de la feuille "Data"
            self.df_brut = read_sql_sheet(
                self.df_path["data"], self.df_path["path"], dtype={self.year_col: str}
            )

            date_parsed = pd.to_datetime(
                self.df_brut[self.year_col].astype(str).str.strip(), errors="coerce"
            )

            self.df_brut = (
                self.df_brut.assign(_date_sort=date_parsed)
                .dropna(subset=["_date_sort"])
                .sort_values("_date_sort", ascending=True)
                .drop(columns=["_date_sort"])
                .reset_index(drop=True)
            )

            df = get_last_dates(
                self.df_brut, self.year_col, self.value_col, type_data=self.type_data
            )

            # Appliquer la transformation des log-variations
            df_log_var = log_variation(
                df,
                self.value_col,
                self.year_col,
                name="log_variation",
                type_data=self.type_data,
            )

            if self.ajustement_var is not None:
                try:
                    # Copier et renommer la colonne de date de l'ajustement pour correspondre à celle du modèle actuel
                    ajustement_df = self.ajustement_var.df.copy()
                    if self.ajustement_var.year_col != self.year_col:
                        ajustement_df = ajustement_df.rename(
                            columns={self.ajustement_var.year_col: self.year_col}
                        )

                    df_net = pd.merge(
                        df_log_var,
                        ajustement_df.rename(
                            columns={"log_variation": "log_variation_ajust"}
                        ),
                        on=self.year_col,
                        how="inner",
                    )
                    df_net["log_variation_net"] = (
                        df_net["log_variation"] - df_net["log_variation_ajust"]
                    )
                    df_log_var = (df_net[[self.year_col, "log_variation_net"]]).rename(
                        columns={"log_variation_net": "log_variation"}
                    )
                except Exception as e:
                    print(f"Erreur sur la variable d'ajustement (ex: inflation) : {e}")
            # Tronquer les données après start_year
            df_nominal = df_log_var[[self.year_col, "log_variation"]]

            # Copier et renommer la colonne de date de l'inflation pour correspondre à celle du modèle actuel
            # (copie pour éviter de modifier l'objet original qui pourrait être partagé)
            inflate_df = self.inflate.df.copy()
            if self.inflate.year_col != self.year_col:
                inflate_df = inflate_df.rename(
                    columns={self.inflate.year_col: self.year_col}
                )

            self.df = pd.merge(
                df_nominal,
                inflate_df.rename(columns={"log_variation": "log_variation_lat"}),
                on=self.year_col,
                how="inner",
            )
            # Tronquer les données à partir de start_year (end_year est INCLUSIF)
            df_tronc = self.df[
                (self.df[self.year_col] >= self.start_year)
                & (self.df[self.year_col] <= self.end_year)
            ]
            log_var_series = df_tronc.set_index(self.year_col)["log_variation"]
            log_var_inflate = df_tronc.set_index(self.year_col)["log_variation_lat"]
            # Création des séries temporelles décalées
            self.m_t = log_var_series[1:].reset_index(drop=True)
            q_t = log_var_inflate[1:].reset_index(drop=True)
            self.m_tm1 = log_var_series[:-1].reset_index(drop=True)
            self.q_tm1 = log_var_inflate[:-1].reset_index(drop=True)
            # Fusionner m_t et q_t en une matrice 2D (chaque ligne est un vecteur d'état)
            self.x_t = np.column_stack((self.m_t, q_t))
            # Fusionner les états précédents m_tm1 et q_tm1 de la même façon
            self.x_tm1 = np.column_stack((self.m_tm1, self.q_tm1))
            self.z0 = df_tronc["log_variation"].iloc[-1]
        except Exception as e:
            print(f"Erreur lors de la préparation des données : {e}")

    def _neg_log_likelihood(self, params):
        """
        Calcule la négative log-vraisemblance pour le modèle en fonction du vecteur de paramètres.

        params : list ou array de 5 éléments [mu, kappa, alpha, sigma, rho]
        """
        mu, kappa, alpha, sigma, rho = params.tolist()

        sigma_epsilon, cov_epsilon = self._compute_vol_and_corr_sigma(
            kappa, sigma, alpha, rho
        )

        # Calcul du coefficient b
        b = (alpha * self.inflate.kappa) / (self.inflate.kappa - kappa)

        # Calcul du déterminant de la matrice de covariance
        det_cov = sigma_epsilon**2 * self.inflate.sigma_epsilon**2 - cov_epsilon**2

        # Calcul de l'inverse de la matrice de covariance (formule explicite)
        inv_cov = (1 / det_cov) * np.array(
            [
                [self.inflate.sigma_epsilon**2, -cov_epsilon],
                [-cov_epsilon, sigma_epsilon**2],
            ]
        )

        # Calcul des coefficients autoregressifs (ar_coeff → 0 pour ABM, e^{-κδ} pour OU)
        exp_m = np.exp(-kappa)  # κ_m > 0 toujours
        exp_q = ar_coeff(self.inflate.kappa, 1.0)  # 0 si κ_q=0 (ABM)

        # Calcul des différents termes pour le vecteur de constante c_t
        term1 = Psi(kappa, 1.0) * (
            kappa * mu + alpha * self.inflate.kappa * self.inflate.mu
        )
        term2 = (
            b
            * (Psi(self.inflate.kappa, 1.0) - Psi(kappa, 1.0))
            * self.inflate.kappa
            * self.inflate.mu
        )

        c_t = np.array([term1 + term2, (1 - exp_q) * self.inflate.mu])

        # Définition de la matrice d'évolution phi_t
        phi_t = np.array([[exp_m, b * (exp_q - exp_m)], [0, exp_q]])

        # Calcul de la prédiction pour chaque observation :
        # x_pred = c_t + phi_t @ x_tm1
        # Comme x_tm1 est de dimension (n, 2), on transpose pour effectuer la multiplication
        x_pred = c_t + (phi_t @ self.x_tm1.T).T

        # Calcul de l'erreur pour chaque observation
        errors = self.x_t - x_pred

        # Pour chaque vecteur d'erreur e, calcul de e.T @ inv_cov @ e
        likelihood_terms = np.mean(
            np.array([e.T @ inv_cov @ e for e in errors])
        ) + np.log(det_cov)
        # Retourner la somme des contributions (log-vraisemblance globale)
        return likelihood_terms

    def fit_model(self, df_path, ajustement_var: Optional["Ornstein_Uhlenbeck"] = None):
        """
        Estime directement par vraisemblance les paramètres du modèle :
          - mu   : μ^(m)
          - kappa: κ^(m)
          - alpha: α^(m)
          - sigma_epsilon: σ

        La fonction de négative log-vraisemblance est définie dans self._neg_log_likelihood.
        """

        self.df_path = df_path
        self.ajustement_var = ajustement_var
        self._prepare_data()
        try:
            # Estimation initiale choisie arbitrairement
            initial_guess = np.array(
                [0.1, 0.5, 1, 0.1, 0.5]
            )  # [mu, kappa, alpha, sigma, rho]

            # Bornes :
            # - mu : non borné
            # - kappa : > 0
            # - alpha : non borné
            # - sigma : > 0
            # - rho : >= -1 et <=1
            bounds = [(None, None), (1e-6, None), (None, None), (1e-6, None), (-1, 1)]

            # Optimisation directe de la négative log-vraisemblance
            result = minimize(self._neg_log_likelihood, initial_guess, bounds=bounds)
            if not result.success:
                print("L'optimisation ML n'a pas convergé.")

            # Extraction des paramètres optimisés
            self.mu, self.kappa, self.alpha, self.sigma, self.rho = result.x
            self.compute_residuals()
        except Exception as e:
            print(f"Erreur lors de l'estimation ML : {e}")

    def _compute_vol_and_corr_sigma(self, kappa, sigma, alpha, rho, delta=1):
        """
        Calcule sigma^{(m)2}_{n} et sigma^{(mq)2}_{n} selon les équations données.

        Paramètres :
        - kappa_m, kappa_q : indices pour la fonction compute_K
        - sigma_m, sigma_q : écarts types
        - alpha_m : coefficient alpha
        - rho_mq : corrélation entre m et q
        - b : paramètre scalaire
        - compute_K : fonction qui calcule K_{ij}

        Retourne :
        - La valeur de sigma^{(m)2}_{n}
        - La valeur de sigma^{(mq)2}_{n}
        """

        b = (alpha * self.inflate.kappa) / (self.inflate.kappa - kappa)
        # Calcul des termes K
        K_mm = compute_K(kappa, kappa, delta)
        K_mq = compute_K(kappa, self.inflate.kappa, delta)
        K_qq = compute_K(self.inflate.kappa, self.inflate.kappa, delta)

        # Premier grand terme
        term1 = K_mm * (
            sigma**2
            + 2 * sigma * self.inflate.sigma * (alpha - b) * rho
            + (alpha - b) ** 2 * self.inflate.sigma**2
        )

        # Deuxième grand terme
        term2 = (
            2
            * b
            * K_mq
            * (sigma * self.inflate.sigma * rho + (alpha - b) * self.inflate.sigma**2)
        )

        # Troisième grand terme
        term3 = b**2 * K_qq * self.inflate.sigma**2

        sigma_epsilon = np.sqrt(term1 + term2 + term3)
        # Calcul de \sigma^{(mq)2}_{n}
        term_4 = sigma * self.inflate.sigma * K_mq * rho
        term_5 = (self.inflate.sigma**2) * (alpha * K_mq + b * (K_qq - K_mq))

        cov_epsilon = term_4 + term_5

        return sigma_epsilon, cov_epsilon

    def compute_residuals(self):
        """
        Calcule les résidus du modèle comme la différence entre les valeurs observées m_t
        et les valeurs prédites m_pred.
        """

        self.sigma_epsilon, cov_epsilon = self._compute_vol_and_corr_sigma(
            self.kappa, self.alpha, self.sigma, self.rho
        )
        self.correlation_epsilon = cov_epsilon / (
                self.sigma_epsilon * self.inflate.sigma_epsilon
        )
        self.log_lik = -0.5 * (
                self._neg_log_likelihood(params=np.array([self.mu, self.kappa, self.alpha, self.sigma, self.rho])) + np.log(2 * np.pi)
        )  # La log-vraisemblance est l'opposé de la valeur minimisée
        try:
            # Calcul du coefficient b
            b = (self.alpha * self.inflate.kappa) / (self.inflate.kappa - self.kappa)

            # Calcul des exponentielles
            exp_m = np.exp(-self.kappa)  # κ_m > 0 toujours
            exp_q = ar_coeff(self.inflate.kappa, 1.0)  # 0 si κ_q=0 (ABM)

            # Calcul des termes du modèle
            term1 = exp_m * self.m_tm1
            term2 = b * (exp_q - exp_m) * self.q_tm1
            term3 = Psi(self.kappa, 1.0) * (
                self.kappa * self.mu + self.alpha * self.inflate.kappa * self.inflate.mu
            )
            term4 = (
                b
                * (Psi(self.inflate.kappa, 1.0) - Psi(self.kappa, 1.0))
                * self.inflate.kappa
                * self.inflate.mu
            )

            # Prédiction de m_t selon le modèle
            m_pred = term1 + term2 + term3 + term4

            df_tronc = self.df[
                (self.df[self.year_col] >= self.start_year)
                & (self.df[self.year_col] <= self.end_year)
            ]  # Appliquer le filtre (end_year est INCLUSIF)

            dates = (
                df_tronc[self.year_col].iloc[1:].reset_index(drop=True)
            )  # Aligner avec m_t
            # Calcul des résidus
            residuals = self.m_t - m_pred
            self.sw_test = stats.shapiro(residuals / self.sigma_epsilon)
            self.r_squared = r2_score(self.m_t, m_pred)
            self.r_pseudo_squared = 1 - (self.log_lik / log_likelihood_null(self.m_t))
            # Vérifier la correspondance des tailles
            if len(dates) != len(residuals):
                raise ValueError(
                    f"Taille non correspondante : {len(dates)} dates et {len(residuals)} résidus"
                )

            # Créer le DataFrame des résidus standardisés sans index
            self.residuals = pd.DataFrame(
                {
                    "Date": dates,
                    "Residuals": residuals,  # S'assurer qu'il n'y a pas d'index
                }
            )

            return self.residuals

        except Exception as e:
            print(f"Erreur lors du calcul des résidus standardisés : {e}")
            return None

    def simulate(
        self,
        m0,
        q0,
        start_date,
        df_mu_q_path=None,
        T=None,
        N=5,
        delta=1,
        seed=None,
        q_sim_noise_in: list[np.ndarray] = None,
    ):
        """
        Simule N trajectoires du processus m_t (modèle Vasicek à deux facteurs)
        à partir d'une simulation de q_t (q_sim) et selon la formule :

        Les dates sont gérées sous forme de datetime et start_date doit être fourni.

        Paramètres :
          - m0         : condition initiale pour m_t (valeur unique).
          - q_sim      : tableau numpy de dimensions (T+1, N) correspondant à la simulation de q_t.
          - start_date : datetime de départ (ex : pd.to_datetime('2020-01-01')).
          - df_mu_path : chemin vers le fichier Excel (optionnel). Dans ce cas, on utilise la colonne 'Date'
                         et la variable référencée par 'Variable_m' dans le sheet "Index".
          - T          : nombre d'étapes à simuler. Si df_mu_path est fourni, T = len(df) (ou min(T, len(df))).
                         Si df_mu_path est None, T doit être fourni.
          - N          : nombre de trajectoires à simuler.
          - delta      : dans le cas constant (sans df_mu_path), pas de temps en années (obligatoire).
          - seed       : graine aléatoire pour la reproductibilité (optionnel).

        Retourne :
          Un DataFrame avec T+1 lignes, incluant :
             - une colonne 'Date' (datetime),
             - une colonne 'mu_m' (la trajectoire de μ^(m) utilisée),
             - une colonne par trajectoire simulée (m_simul_1, m_simul_2, …).
        """
        if q_sim_noise_in is None:
            q_sim_noise_in = [None] * 3

        # Conversion de start_date en datetime
        start_date = pd.to_datetime(start_date)
        if seed is not None:
            np.random.seed(seed)

        deltas, noise_in, q_sim_brut = q_sim_noise_in
        q_noise = None
        if noise_in is None:
            q_sim_brut, q_noise, deltas = self.inflate.simulate(
                z0=q0,
                start_date=start_date,
                df_mu_path=df_mu_q_path,
                T=T,
                N=N,
                delta=delta,
                seed=seed,
            )
        q_sim = q_sim_brut.drop(columns=["Date", "mu"]).values

        T, N = q_sim.shape[0] - 1, q_sim.shape[1]

        m = np.zeros((T + 1, N))
        m[0, :] = m0

        if noise_in is not None and noise_in.shape != (T, N):
            raise ValueError(f"Dimension {noise_in.shape} != attendues ({T}, {N})")
        epsilon = np.random.normal(0, 1, (T, N)) if noise_in is None else noise_in

        b = (self.alpha * self.inflate.kappa) / (self.inflate.kappa - self.kappa)
        mu_q = q_sim_brut.loc[:, "mu"].values
        for i in range(1, T + 1):
            d = deltas[i - 1]
            exp_m = np.exp(-self.kappa * d)  # κ_m > 0 toujours
            exp_q = ar_coeff(self.inflate.kappa, d)  # 0 si κ_q=0 (ABM)
            sigma_epsilon, cov_epsilon = self._compute_vol_and_corr_sigma(
                kappa=self.kappa,
                sigma=self.sigma,
                alpha=self.alpha,
                rho=self.rho,
                delta=d,
            )
            # Calcul des différents termes pour le vecteur de constante c_t
            term1 = Psi(self.kappa, d) * (
                self.kappa * self.mu + self.alpha * self.inflate.kappa * mu_q[i]
            )

            term2 = (
                b
                * (Psi(self.inflate.kappa, d) - Psi(self.kappa, d))
                * self.inflate.kappa
                * mu_q[i]
            )

            if noise_in is None:
                q_sigma_epsilon = self.inflate.sigma * np.sqrt(
                    Psi(2 * self.inflate.kappa, d)
                )
                noise = (
                    cov_epsilon / q_sigma_epsilon * q_noise[i - 1, :]
                    + np.sqrt(sigma_epsilon**2 - (cov_epsilon / q_sigma_epsilon) ** 2)
                    * epsilon[i - 1, :]
                )
            else:
                noise = sigma_epsilon * noise_in[i - 1, :]

            m[i, :] = (
                exp_m * m[i - 1, :]
                + b * (exp_q - exp_m) * q_sim[i - 1, :]
                + term1
                + term2
                + noise
            )
        # Créer le DataFrame initial avec la colonne 'Date'
        m_sim = q_sim_brut[["Date"]].copy()

        # Construire un dictionnaire contenant les colonnes à ajouter
        new_columns = {f"m_simul_{p + 1}": m[:, p] for p in range(N)}

        # Créer un DataFrame pour ces nouvelles colonnes en utilisant le même index que m_sim
        df_new = pd.DataFrame(new_columns, index=m_sim.index)

        # Concaténer en une seule opération
        m_sim = pd.concat([m_sim, df_new], axis=1)
        return m_sim, q_sim_brut

    def summary(self):
        """
        Affiche un résumé des paramètres estimés.
        """
        self.compute_residuals()
        print(f"\nR² : {self.r_squared:.4f}")
        print(f"pvalue Shapiro-Wilk : {self.sw_test.pvalue:.4f}")
        print("Estimation par Maximum de Vraisemblance:")
        print(f"Alpha : {self.alpha:.4f}")
        print(f"sigma_epsilon : {self.sigma_epsilon:.4f}")
        print(f"sigma : {self.sigma:.4f}")
        print(f"Kappa : {self.kappa:.4f}")
        print(f"Mu    : {self.mu:.4f}")
        print(f"rho_epsilon    : {self.correlation_epsilon:.4f}")
        print(f"rho    : {self.rho:.4f}")
        print(f"Log-vraisemblance : {self.log_lik:.4f}")

    def get_results(self):
        """
        Retourne un dictionnaire contenant les paramètres estimés.
        """
        self.compute_residuals()
        return {
            "alpha": self.alpha,
            "sigma_epsilon": self.sigma_epsilon,
            "kappa": self.kappa,
            "mu": self.mu,
            "r_squared": self.r_squared,
            "r_pseudo_squared": self.r_pseudo_squared,
            "log_lik": self.log_lik,
            "Shapiro-Wilk": self.sw_test,
        }
