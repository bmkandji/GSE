from gse_engine.ahlgrim.tools import *
from scipy.optimize import minimize, minimize_scalar
from typing import Optional
from .ornstein_ulhenbeck import Ornstein_Uhlenbeck
from scipy import stats
from gse_engine.db import read_sql_sheet


class Two_factor_Vasicek:
    def __init__(self, taux_long_model: Ornstein_Uhlenbeck, params: list = None):
        """
        Initialise le modèle OU et estime par maximum de vraisemblance les paramètres
        d'un modèle AR(1) appliqué aux log-variations extraites des données.

        Les paramètres estimés sont :
            - φ (phi) : coefficient AR(1)
            - σ (sigma) : écart-type de l'erreur
            - κ = -log(φ) (kappa)
            - μ = α/(1-φ) (mu)
            - la log-vraisemblance (log_lik)
            - le R² de la régression
        """
        self._type = "Va2"
        self.df_path = {"path": None, "index": "Index"}
        self.taux_long = taux_long_model
        self.df = None
        self.df_brut = None
        self.ajustement_var = None
        self.year_col = None
        self.value_col = None
        self.type_data = None
        self.start_year = None
        self.end_year = None
        if params is None:
            params = [None] * 3
        # Paramètres estimés
        self.kappa, self.sigma, self.rho = params
        self.sigma_epsilon = None
        self.residuals = None  # Stocke les résidus
        self.sw_test = None
        self.correlation_epsilon_rl = None
        self.log_lik = None  # log-vraisemblance
        self.r_squared = None
        self.r_pseudo_squared = None
        self.prime = 0
        self.z0 = None

        # Séries de log-variations
        self.r_t = None
        self.r_tm1 = None
        self.l_tm1 = None
        self.x_t = None
        self.x_tm1 = None

    def _prepare_data(self):
        """Tronquer les données à partir de start_year et préparer les séries pour la régression."""
        try:
            # Charger les métadonnées
            # dico = pd.read_excel(self.df_path["path"], sheet_name=self.df_path["index"])
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

            # Vérifier que le taux_long a bien été calibré et a un df
            if self.taux_long.df is None or self.taux_long.df.empty:
                raise ValueError(
                    "Le modèle taux_long (latent) n'a pas été calibré correctement"
                )

            # Renommer la colonne de date du taux_long pour correspondre à celle du VA2
            taux_long_df = self.taux_long.df.copy()
            if self.taux_long.year_col != self.year_col:
                taux_long_df = taux_long_df.rename(
                    columns={self.taux_long.year_col: self.year_col}
                )

            # Renommer la colonne log_variation en log_variation_lat
            taux_long_df = taux_long_df.rename(
                columns={"log_variation": "log_variation_lat"}
            )

            self.df = pd.merge(
                df_nominal,
                taux_long_df,
                on=self.year_col,
                how="inner",
            )

            if self.df.empty:
                raise ValueError(
                    f"Aucune date commune trouvée entre le VA2 et le taux_long. Vérifier les périodes de calibration."
                )

            # Tronquer les données à partir de start_year (end_year est INCLUSIF)
            df_tronc = self.df[
                (self.df[self.year_col] >= self.start_year)
                & (self.df[self.year_col] <= self.end_year)
            ]
            log_var_series = df_tronc.set_index(self.year_col)["log_variation"]
            log_var_taux_long = df_tronc.set_index(self.year_col)["log_variation_lat"]
            # Création des séries temporelles décalées
            self.r_t = log_var_series[1:].reset_index(drop=True)
            l_t = log_var_taux_long[1:].reset_index(drop=True)
            self.r_tm1 = log_var_series[:-1].reset_index(drop=True)
            self.l_tm1 = log_var_taux_long[:-1].reset_index(drop=True)
            # Fusionner r_t et l_t en une matrice 2D (chaque ligne est un vecteur d'état)
            self.x_t = np.column_stack((self.r_t, l_t))
            # Fusionner les états précédents r_tm1 et l_tm1 de la même façon
            self.x_tm1 = np.column_stack((self.r_tm1, self.l_tm1))
            self.z0 = df_tronc["log_variation"].iloc[-1]
        except Exception as e:
            print(f"Erreur lors de la préparation des données : {e}")

    def _neg_log_likelihood(self, params):
        """
        Calcule la négative log-vraisemblance pour le modèle en fonction du vecteur de paramètres.

        params : list ou array de 5 éléments [kappa, sigma, rho]
        """

        kappa, sigma, rho = params.tolist()

        sigma_epsilon, cov_epsilon = self._compute_vol_and_corr_sigma(kappa, sigma, rho)

        # Calcul du coefficient b
        a = kappa / (kappa - self.taux_long.kappa)

        # Calcul du déterminant de la matrice de covariance
        det_cov = sigma_epsilon**2 * self.taux_long.sigma_epsilon**2 - cov_epsilon**2

        # Calcul de l'inverse de la matrice de covariance (formule explicite)
        inv_cov = (1 / det_cov) * np.array(
            [
                [self.taux_long.sigma_epsilon**2, -cov_epsilon],
                [-cov_epsilon, sigma_epsilon**2],
            ]
        )

        # Calcul des exponentielles
        exp_r = np.exp(-kappa)
        exp_l = np.exp(-self.taux_long.kappa)
        delta = 1.0  # pas de calibration annuel

        # Calcul des différents termes pour le vecteur de constante c_t
        # Utilisation de Ψ pour cohérence avec le papier
        psi_r = Psi(kappa, delta)
        psi_l = Psi(self.taux_long.kappa, delta)
        gam_l = gamma_coeff(self.taux_long.kappa)

        term_cst = (
            ((kappa * self.taux_long.kappa) / (kappa - self.taux_long.kappa))
            * (psi_l - psi_r)
            * self.taux_long.mu
        )

        c_t = np.array([term_cst, gam_l * psi_l * self.taux_long.mu])

        # Définition de la matrice d'évolution phi_t
        phi_t = np.array([[exp_r, a * (exp_l - exp_r)], [0, exp_l]])

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
          - kappa: κ^(r)
          - sigma_epsilon: σ
          - rho_epsilon

        La fonction de négative log-vraisemblance est définie dans self._neg_log_likelihood.
        """
        self.df_path = df_path
        self.ajustement_var = ajustement_var
        self._prepare_data()
        try:
            # Estimation initiale choisie arbitrairement
            initial_guess = np.array([0.1, 0.5, 0.5])  # [kappa, sigma_epsilon, rho]

            # Bornes :
            # - kappa : > 0
            # - sigma_epsilon : > 0
            # - rho : >= -1 et <=1
            bounds = [(1e-6, None), (1e-6, None), (-1, 1)]

            # Optimisation directe de la négative log-vraisemblance
            result = minimize(self._neg_log_likelihood, initial_guess, bounds=bounds)
            if not result.success:
                print("L'optimisation ML n'a pas convergé.")

            # Extraction des paramètres optimisés
            self.kappa, self.sigma, self.rho = result.x
            self.compute_residuals()

        except Exception as e:
            print(f"Erreur lors de l'estimation ML : {e}")

    def _compute_vol_and_corr_sigma(
        self, kappa: float, sigma: float, rho: float, delta: float = 1
    ):
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

        a = kappa / (kappa - self.taux_long.kappa)

        # Calcul des termes K
        K_rr = compute_K(kappa, kappa, delta)
        K_rl = compute_K(kappa, self.taux_long.kappa, delta)
        K_ll = compute_K(self.taux_long.kappa, self.taux_long.kappa, delta)

        sigma_epsilon = np.sqrt(
            (sigma**2) * K_rr
            + 2 * a * sigma * self.taux_long.sigma * rho * (K_rl - K_rr)
            + a**2 * (self.taux_long.sigma**2) * (K_ll - 2 * K_rl + K_rr)
        )

        cov_epsilon = sigma * self.taux_long.sigma * K_rl * rho + a * (
            self.taux_long.sigma**2
        ) * (K_ll - K_rl)

        return sigma_epsilon, cov_epsilon

    def compute_residuals(self):
        """
        Calcule les résidus du modèle comme la différence entre les valeurs observées m_t
        et les valeurs prédites m_pred.
        """

        self.sigma_epsilon, cov_epsilon = self._compute_vol_and_corr_sigma(
            self.kappa, self.sigma, self.rho
        )
        self.correlation_epsilon_rl = cov_epsilon / (
                self.sigma_epsilon * self.taux_long.sigma_epsilon
        )
        self.log_lik = -0.5 * (
                self._neg_log_likelihood(params=np.array([self.kappa, self.sigma, self.rho])) + np.log(2 * np.pi)
        )  # La log-vraisemblance est l'opposé de la valeur minimisée
        try:
            # Calcul du coefficient b
            a = self.kappa / (self.kappa - self.taux_long.kappa)

            # Calcul des exponentielles
            exp_r = np.exp(-self.kappa)
            exp_l = np.exp(-self.taux_long.kappa)
            delta = 1.0

            # Utilisation de Ψ pour cohérence
            psi_r = Psi(self.kappa, delta)
            psi_l = Psi(self.taux_long.kappa, delta)

            # Calcul des termes du modèle
            term1 = exp_r * self.r_tm1
            term2 = a * (exp_l - exp_r) * self.l_tm1
            term3 = (
                (
                    (self.kappa * self.taux_long.kappa)
                    / (self.kappa - self.taux_long.kappa)
                )
                * (psi_l - psi_r)
                * self.taux_long.mu
            )

            # Prédiction de r_t selon le modèle
            r_pred = term1 + term2 + term3

            df_tronc = self.df[
                (self.df[self.year_col] >= self.start_year)
                & (self.df[self.year_col] <= self.end_year)
            ]  # Appliquer le filtre (end_year est INCLUSIF)
            dates = (
                df_tronc[self.year_col].iloc[1:].reset_index(drop=True)
            )  # Aligner avec r_t
            # Calcul des résidus
            residuals = self.r_t - r_pred
            self.sw_test = stats.shapiro(residuals / self.sigma_epsilon)
            self.r_squared = r2_score(self.r_t, r_pred)
            self.r_pseudo_squared = 1 - (self.log_lik / log_likelihood_null(self.r_t))
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

        except Exception as e:
            print(f"Erreur lors du calcul des résidus standardisés : {e}")
            return None

    def simulate(
        self,
        r0,
        l0,
        start_date,
        df_mu_path=None,
        T=None,
        N=1,
        delta=1,
        seed=None,
        l_sim_noise_in: list[np.ndarray] = None,
    ):
        """
        Simule N trajectoires du processus r_t (modèle Vasicek à deux facteurs)
        à partir d'une simulation de l_t (l_sim) et selon la formule :

            r_{t_n} = e^{-δ_n κ^(r)} r_{t_{n-1}}
                      + a (e^{-δ_n κ^(l)} - e^{-δ_n κ^(r)}) l_{t_{n-1}}
                      + (1/(κ^(r)-κ^(l))) [κ^(r)(1-e^{-δ_n κ^(l)}) - κ^(l)(1-e^{-δ_n κ^(r)})] μ^(r)_t
                      + σ^(r)_n ε^(r)_n,

        où
            a = κ^(r)/(κ^(r)-κ^(l)),
        et où σ^(r)_n (ainsi que la covariance associée) est calculé par
            self._compute_vol_and_corr_sigma(kappa=self.kappa_r, sigma=self.sigma_r, rho=self.rho_rl)

        Les dates sont gérées sous forme de datetime et start_date doit être fourni.

        Paramètres :
          - r0         : condition initiale pour r_t (valeur unique).
          - l_sim      : tableau numpy de dimensions (T+1, N) correspondant à la simulation de l_t.
          - start_date : datetime de départ (ex : pd.to_datetime('2020-01-01')).
          - df_mu_path : chemin vers le fichier Excel (optionnel). Dans ce cas, on utilise la colonne 'Date'
                         et la variable référencée par 'Variable_r' dans le sheet "Index".
          - T          : nombre d'étapes à simuler. Si df_mu_path est fourni, T = len(df) (ou min(T, len(df))).
                         Si df_mu_path est None, T doit être fourni.
          - N          : nombre de trajectoires à simuler.
          - delta      : dans le cas constant (sans df_mu_path), pas de temps en années (obligatoire).
          - seed       : graine aléatoire pour la reproductibilité (optionnel).

        Retourne :
          Un DataFrame avec T+1 lignes, incluant :
             - une colonne 'Date' (datetime),
             - une colonne 'mu_r' (la trajectoire de μ^(r) utilisée),
             - une colonne par trajectoire simulée (r_simul_1, r_simul_2, …).
        """
        if l_sim_noise_in is None:
            l_sim_noise_in = [None] * 3
        deltas, noise_in, l_sim_brut = l_sim_noise_in
        l_noise = None
        if noise_in is None:
            l_sim_brut, l_noise, deltas = self.taux_long.simulate(
                z0=l0,
                start_date=start_date,
                df_mu_path=df_mu_path,
                T=T,
                N=N,
                delta=delta,
                seed=seed,
            )
        l_sim = l_sim_brut.drop(columns=["Date", "mu"]).values
        T, N = l_sim.shape[0] - 1, l_sim.shape[1]

        r = np.zeros((T + 1, N))
        r[0, :] = r0

        if noise_in is not None and noise_in.shape != (T, N):
            raise ValueError(f"Dimension {noise_in.shape} != attendues ({T}, {N})")
        epsilon = np.random.normal(0, 1, (T, N)) if noise_in is None else noise_in

        a = self.kappa / (self.kappa - self.taux_long.kappa)
        mu = l_sim_brut.loc[:, "mu"].values
        for i in range(1, T + 1):
            d = deltas[i - 1]
            exp_r = np.exp(-self.kappa * d)
            exp_l = np.exp(-self.taux_long.kappa * d)
            psi_r = Psi(self.kappa, d)
            psi_l = Psi(self.taux_long.kappa, d)
            sigma_epsilon, cov_epsilon = self._compute_vol_and_corr_sigma(
                kappa=self.kappa, sigma=self.sigma, rho=self.rho, delta=d
            )
            if noise_in is None:
                l_sigma_epsilon = self.taux_long.sigma * np.sqrt(
                    Psi(2.0 * self.taux_long.kappa, d)
                )
                noise = (
                    cov_epsilon / l_sigma_epsilon * l_noise[i - 1, :]
                    + np.sqrt(sigma_epsilon**2 - (cov_epsilon / l_sigma_epsilon) ** 2)
                    * epsilon[i - 1, :]
                )
            else:
                noise = sigma_epsilon * noise_in[i - 1, :]
            r[i, :] = (
                exp_r * r[i - 1, :]
                + a * (exp_l - exp_r) * l_sim[i - 1, :]
                + (
                    (self.kappa * self.taux_long.kappa)
                    / (self.kappa - self.taux_long.kappa)
                )
                * (psi_l - psi_r)
                * mu[i]
                + noise
            )
        # Créer le DataFrame initial avec la colonne 'Date'
        r_sim = l_sim_brut[["Date"]].copy()

        # Créer un dictionnaire contenant les colonnes à ajouter
        new_columns = {f"r_simul_{p + 1}": r[:, p] for p in range(N)}

        # Créer un DataFrame pour ces nouvelles colonnes en utilisant le même index que r_sim
        df_new = pd.DataFrame(new_columns, index=r_sim.index)

        # Concaténer en une seule opération
        r_sim = pd.concat([r_sim, df_new], axis=1)
        return r_sim, l_sim_brut

    def compute_primes(self, zc_path: str, sep=","):
        term_rates = importer_et_fusionner_csv(zc_path, sep=sep)
        liste_tr = transformer_en_liste_df(term_rates)

        def objectif(lambda_l):
            cumul = 0
            for df_data in liste_tr:
                df = df_data.copy().iloc[1:-1]
                data = [
                    (
                        term_rate(
                            maturity,
                            df["Valeur"].iloc[0],
                            df["Valeur"].iloc[-1],
                            self.kappa,
                            self.taux_long.kappa,
                            self.sigma,
                            self.taux_long.sigma,
                            self.taux_long.mu,
                            self.rho,
                            lambda_l,
                        )
                        - taux
                    )
                    ** 2
                    for maturity, taux in zip(df["Maturite"], df["Valeur"])
                ]
                cumul += np.mean(data)
            return cumul

        # Définir une valeur initiale et des bornes pour lambda_r et lambda_l (strictement positives)
        max_prime = self.taux_long.mu / (self.taux_long.sigma / self.taux_long.kappa)
        if max_prime <= 0:
            self.prime = 0
        else:
            result = minimize_scalar(objectif, bounds=(0, max_prime), method="bounded")

            if result.success:
                self.prime = result.x
            else:
                print("L'optimisation a échoué :", result.message)

    def summary(self):
        """
        Affiche un résumé des paramètres estimés.
        """
        self.compute_residuals()
        print(f"\nR² : {self.r_squared:.4f}")
        print(f"pvalue Shapiro-Wilk : {self.sw_test.pvalue:.4f}")
        print("Estimation par Maximum de Vraisemblance:")
        print(f"sigma_epsilon : {self.sigma_epsilon:.4f}")
        print(f"sigma : {self.sigma:.4f}")
        print(f"Kappa : {self.kappa:.4f}")
        print(f"rho_epsilon    : {self.correlation_epsilon_rl:.4f}")
        print(f"rho    : {self.rho:.4f}")
        print(f"Log-vraisemblance : {self.log_lik:.4f}")

    def get_results(self):
        """
        Retourne un dictionnaire contenant les paramètres estimés.
        """
        self.compute_residuals()
        return {
            "sigma_epsilon": self.sigma_epsilon,
            "rho_epsilon": self.correlation_epsilon_rl,
            "kappa": self.kappa,
            "rho": self.rho,
            "sigma": self.sigma,
            "r_squared": self.r_squared,
            "r_pseudo_squared": self.r_pseudo_squared,
            "log_lik": self.log_lik,
            "Shapiro-Wilk": self.sw_test,
        }
