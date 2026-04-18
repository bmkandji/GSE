from gse_engine.db import read_sql_sheet
from .two_factor_vasicek import Two_factor_Vasicek
from gse_engine.ahlgrim.tools import *
from typing import Union, Dict
from scipy.stats import multivariate_normal
from scipy import stats
import warnings


# ═════════════════════════════════════════════════════════════════════════
#  HARDY UNIVARIÉ — Réécriture complète du calibrage
# ═════════════════════════════════════════════════════════════════════════


class Hardy:
    def __init__(self, params=None, df_path=None, ajustement_var=None):
        self._type = "Hd"
        if df_path is None:
            df_path = {"path": None, "index": "Index"}
        if params is None:
            params = [
                np.array([0.7, 0.3]),
                np.array([30, 15]),
                np.array([[0.9, 0.1], [0.05, 0.95]]),
                np.array([0.5, 0.5]),
            ]
        self.ajustement_var = ajustement_var
        self.df_path = df_path
        self.df = None
        self.df_brut = None
        self.year_col = None
        self.value_col = None
        self.type_data = None
        self.frequence = None
        self.start_year = None
        self.end_year = None
        self.mu, self.sigma, self.P, self.pi = params
        self.delta = None
        self.d = len(self.mu)
        self.observations = None
        self.filter = None
        self.z0 = None

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
            self.frequence = dico.loc[dico["Refrence"] == "frequence", "data"].values[0]
            self.df_path["data"] = dico.loc[
                dico["Refrence"] == "DataLeaf", "data"
            ].values[0]

            self.delta = 0.25 if self.frequence == "quarter" else 1

            if len(self.year_col) == 0 or len(self.value_col) == 0:
                raise ValueError(
                    "Les colonnes pour 'Date' , 'type_data' et 'Variable' n'ont pas été trouvées dans le fichier Index."
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

            # CORRECTION: Filtrer df_brut par la période de calibration AVANT get_last_dates()
            # pour s'assurer que les log-variations sont calculées uniquement sur la période de calibration
            df_brut_filtered = self.df_brut.copy()

            # Conversion robuste de l'année avec gestion des erreurs
            try:
                # Convertir en datetime pour extraire l'année de manière fiable
                if df_brut_filtered[self.year_col].dtype == "object":
                    # Si c'est une string, essayer de la convertir en datetime
                    date_col_parsed = pd.to_datetime(
                        df_brut_filtered[self.year_col], errors="coerce"
                    )
                    df_brut_filtered["year_extracted"] = date_col_parsed.dt.year
                else:
                    # Si c'est déjà un datetime ou un entier
                    try:
                        df_brut_filtered["year_extracted"] = pd.to_datetime(
                            df_brut_filtered[self.year_col], errors="coerce"
                        ).dt.year
                    except:
                        # Dernier recours: extraire les 4 premiers caractères si c'est au format 'YYYY-...'
                        year_str = (
                            df_brut_filtered[self.year_col].astype(str).str.strip()
                        )
                        df_brut_filtered["year_extracted"] = year_str.str[:4].astype(
                            int
                        )

                # Supprimer les lignes avec années invalides (NaN)
                df_brut_filtered = df_brut_filtered.dropna(subset=["year_extracted"])
                df_brut_filtered["year_extracted"] = df_brut_filtered[
                    "year_extracted"
                ].astype(int)

                # Filtrer par période de calibration
                df_brut_filtered = df_brut_filtered[
                    (df_brut_filtered["year_extracted"] >= self.start_year)
                    & (df_brut_filtered["year_extracted"] <= self.end_year)
                ].drop(columns=["year_extracted"])

            except Exception as e:
                print(f"Erreur lors du filtrage par période de calibration : {e}")
                import traceback

                traceback.print_exc()
                # En cas d'erreur, utiliser toutes les données (comportement de fallback)
                df_brut_filtered = self.df_brut.copy()

            df = get_last_dates(
                df_brut_filtered,
                self.year_col,
                self.value_col,
                output=self.frequence,
                type_data=self.type_data,
            )
            self.delta = 1

            # Appliquer la transformation des log-variations
            df_log_var = log_variation(
                df, self.value_col, self.year_col, type_data=self.type_data
            )

            if self.ajustement_var is not None and self.ajustement_var.df_brut is not None:
                try:
                    ajustement_var = get_last_dates(
                        self.ajustement_var.df_brut,
                        self.ajustement_var.year_col,
                        self.ajustement_var.value_col,
                        output=self.frequence,
                        type_data=self.ajustement_var.type_data,
                    )
                    variation_ajustement_var = log_variation(
                        ajustement_var,
                        self.ajustement_var.value_col,
                        self.ajustement_var.year_col,
                        type_data=self.ajustement_var.type_data,
                    )
                    # Renommer la colonne de date de l'ajustement pour correspondre à celle du modèle actuel
                    if self.ajustement_var.year_col != self.year_col:
                        variation_ajustement_var = variation_ajustement_var.rename(
                            columns={self.ajustement_var.year_col: self.year_col}
                        )

                    df_log_var.rename(
                        columns={"log_variation": "log_variation_brute"}, inplace=True
                    )
                    df_net = pd.merge(
                        df_log_var,
                        variation_ajustement_var,
                        on=self.year_col,
                        how="inner",
                    )
                    df_net["log_variation_net"] = (
                        df_net["log_variation_brute"] - df_net["log_variation"]
                    )
                    df_log_var = (df_net[[self.year_col, "log_variation_net"]]).rename(
                        columns={"log_variation_net": "log_variation"}
                    )
                except Exception as e:
                    print(f"Erreur sur la variable d'ajustement (ex: inflation) : {e}")

            # Extraire l'année dans une nouvelle colonne, par exemple 'year_extracted'
            # Conversion robuste avec gestion d'erreur
            try:
                # Convertir en datetime pour extraire l'année de manière fiable
                if df_log_var[self.year_col].dtype == "object":
                    date_col_parsed = pd.to_datetime(
                        df_log_var[self.year_col], errors="coerce", format='mixed', dayfirst=True
                    )
                    df_log_var["year_extracted"] = date_col_parsed.dt.year
                else:
                    try:
                        df_log_var["year_extracted"] = pd.to_datetime(
                            df_log_var[self.year_col], errors="coerce"
                        ).dt.year
                    except:
                        # Dernier recours: extraire les 4 premiers caractères
                        year_str = df_log_var[self.year_col].astype(str).str.strip()
                        df_log_var["year_extracted"] = year_str.str[:4].astype(int)

                # Supprimer les lignes avec années invalides
                df_log_var = df_log_var.dropna(subset=["year_extracted"])
                df_log_var["year_extracted"] = df_log_var["year_extracted"].astype(int)

                # Filtrer le DataFrame en utilisant les bornes numériques sur l'année extraite
                # Note: end_year est maintenant INCLUSIF (<=) au lieu d'EXCLUSIF (<)
                self.df = df_log_var[
                    (df_log_var["year_extracted"] >= self.start_year)
                    & (df_log_var["year_extracted"] <= self.end_year)
                ][[self.year_col, "log_variation"]]
            except Exception as e:
                print(f"Erreur lors de l'extraction de l'année : {e}")
                print(
                    f"Échantillon de valeurs: {df_log_var[self.year_col].head().tolist() if self.year_col in df_log_var.columns else 'colonne non trouvée'}"
                )
                raise

            # Rename the date column to "Date" for consistency across all Hardy models
            self.df = self.df.rename(columns={self.year_col: "Date"})
            self.observations = self.df.set_index("Date")["log_variation"].values
            if len(self.df) > 0:
                self.z0 = self.df["log_variation"].iloc[-1]
            else:
                self.z0 = 0.0

        except Exception as e:
            print(f"Erreur lors de la préparation des données : {e}")
        return self.df, self.delta

    def _normal_pdf(self, z, mean, var):
        """Densité N(z; mean, var)."""
        if var <= 0:
            return 0.0
        return (1.0 / np.sqrt(2.0 * np.pi * var)) * np.exp(-0.5 * (z - mean) ** 2 / var)

    def _emission_probabilities(self, z, mu=None, sigma=None):
        """b_i(z) = φ(z; δ·μ_i, δ·σ_i²)"""
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        b = np.array(
            [
                self._normal_pdf(z, self.delta * mu[i], self.delta * sigma[i] ** 2)
                for i in range(self.d)
            ]
        )
        return b

    # ─────────────────────────────────────────────────────────────────
    #  FORWARD PASS (inchangé dans la logique, on stocke aussi b_t)
    # ─────────────────────────────────────────────────────────────────
    def _compute_filtering(self, mu=None, sigma=None, P=None, pi=None):
        if mu is None:
            mu = self.mu
        if sigma is None:
            sigma = self.sigma
        if P is None:
            P = self.P
        if pi is None:
            pi = self.pi

        d = self.d
        n = len(self.observations)
        pi_pred = np.zeros((n, d))
        pi_filt = np.zeros((n, d))
        scales = np.zeros(n)
        B = np.zeros((n, d))  # stockage des émissions pour backward

        # t = 0
        pi_pred[0] = pi
        B[0] = self._emission_probabilities(self.observations[0], mu, sigma)
        c0 = np.dot(pi_pred[0], B[0])
        scales[0] = c0 if c0 > 1e-300 else 1e-300
        pi_filt[0] = (pi_pred[0] * B[0]) / scales[0]

        # t = 1, …, n-1
        for t in range(1, n):
            pi_pred[t] = P.T @ pi_filt[t - 1]
            B[t] = self._emission_probabilities(self.observations[t], mu, sigma)
            ct = np.dot(pi_pred[t], B[t])
            scales[t] = ct if ct > 1e-300 else 1e-300
            pi_filt[t] = (pi_pred[t] * B[t]) / scales[t]

        return {"pi_pred": pi_pred, "pi_filt": pi_filt, "scales": scales, "B": B}

    # ─────────────────────────────────────────────────────────────────
    #  FORWARD-BACKWARD (Baum-Welch) — remplace le lissage RTS
    # ─────────────────────────────────────────────────────────────────
    def _forward_backward(self, mu, sigma, P, pi):
        """
        Algorithme forward-backward complet (Rabiner 1989, Bishop 2006).

        Retourne:
          gamma[t, i]   = P(ρ_t = i | z_{1:T})       (smoothed marginals)
          xi[t, i, j]   = P(ρ_t = i, ρ_{t+1} = j | z_{1:T})  (smoothed joints)
          log_lik       = log P(z_{1:T})
          filter_result = dict with pi_filt, pi_pred, scales, B
        """
        filt = self._compute_filtering(mu, sigma, P, pi)
        alpha = filt["pi_filt"]  # (n, d) scaled forward variables
        scales = filt["scales"]  # (n,)
        B = filt["B"]  # (n, d) emission probabilities

        n, d = alpha.shape

        # ── Backward pass (scaled) ──
        # β̂_T = 1
        # β̂_t(i) = (1/c_{t+1}) · Σ_j P(i,j) · b_j(z_{t+1}) · β̂_{t+1}(j)
        beta = np.ones((n, d))
        for t in range(n - 2, -1, -1):
            beta[t] = P @ (B[t + 1] * beta[t + 1]) / scales[t + 1]

        # ── Smoothed marginals ──
        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum = np.maximum(gamma_sum, 1e-300)
        gamma = gamma / gamma_sum

        # ── Smoothed joints ──
        # ξ_t(i,j) = α̂_t(i) · P(i,j) · b_j(z_{t+1}) · β̂_{t+1}(j) / c_{t+1}
        xi = np.zeros((n - 1, d, d))
        for t in range(n - 1):
            tmp = (alpha[t][:, None] * P) * (B[t + 1] * beta[t + 1])[None, :]
            tmp /= scales[t + 1]
            s = tmp.sum()
            if s > 0:
                tmp /= s
            xi[t] = tmp

        log_lik = np.sum(np.log(np.maximum(scales, 1e-300)))

        return gamma, xi, log_lik, filt

    # ─────────────────────────────────────────────────────────────────
    #  Initialisation guidée par les données
    # ─────────────────────────────────────────────────────────────────
    def _initialize_from_data(self):
        """Initialise mu/sigma par quantiles des observations."""
        z = self.observations
        n = len(z)
        d = self.d

        if d == 1:
            mu = np.array([np.mean(z) / self.delta])
            sigma = np.array([np.std(z, ddof=0) / np.sqrt(self.delta)])
            # Éviter sigma = 0
            sigma = np.maximum(sigma, 1e-6)
            P = np.array([[1.0]])
            pi = np.array([1.0])
            return mu, sigma, P, pi

        # Pour d ≥ 2 : trier les observations et découper en quantiles
        sorted_z = np.sort(z)
        group_size = n // d

        mu = np.zeros(d)
        sigma = np.zeros(d)

        for i in range(d):
            start = i * group_size
            end = (i + 1) * group_size if i < d - 1 else n
            group = sorted_z[start:end]
            mu[i] = np.mean(group) / self.delta
            sigma[i] = max(np.std(group, ddof=0) / np.sqrt(self.delta), 1e-6)

        pi = np.ones(d) / d
        P = np.full((d, d), 0.05 / max(d - 1, 1))
        np.fill_diagonal(P, 0.95)
        # Normaliser les lignes
        P = P / P.sum(axis=1, keepdims=True)

        return mu, sigma, P, pi

    # ─────────────────────────────────────────────────────────────────
    #  FIT MODEL — EM avec forward-backward + prior Dirichlet sur P
    # ─────────────────────────────────────────────────────────────────
    def fit_model(
        self,
        df_path,
        ajustement_var,
        max_iter=200,
        tol=1e-6,
        dirichlet_diag=None,
        dirichlet_offdiag=None,
    ):
        """
        Calibre le modèle par EM (Baum-Welch).

        dirichlet_diag / dirichlet_offdiag :
            Hyper-paramètres du prior Dirichlet sur chaque ligne de P.
            Par défaut : diag=20, offdiag=1 → E[P_ii] ≈ 0.91 pour d=2.
            Mettre None pour désactiver le prior.
        """
        self.df_path = df_path
        self.ajustement_var = ajustement_var
        self._prepare_data()

        n = len(self.observations)
        d = self.d

        # ── Initialisation à partir des données ──
        mu, sigma, P, pi = self._initialize_from_data()

        # ── Prior Dirichlet ──
        if dirichlet_diag is None:
            dirichlet_diag = 20.0
        if dirichlet_offdiag is None:
            dirichlet_offdiag = 1.0

        # Matrice des pseudo-comptes : alpha_prior[i,j]
        alpha_prior = np.full((d, d), dirichlet_offdiag)
        np.fill_diagonal(alpha_prior, dirichlet_diag)

        prev_ll = -np.inf

        for iteration in range(max_iter):

            # === E-step : forward-backward ===
            gamma, xi, log_lik, filt = self._forward_backward(mu, sigma, P, pi)

            # === M-step ===
            mu_new = np.zeros(d)
            sigma_new = np.zeros(d)

            for i in range(d):
                w = gamma[:, i]  # (n,)
                W = w.sum()

                if W < 1e-10:
                    # Régime vide → garder les params précédents
                    mu_new[i] = mu[i]
                    sigma_new[i] = sigma[i]
                    continue

                # μ̂_i = Σ_t z_t γ_t(i) / (δ · Σ_t γ_t(i))
                m_est = np.dot(w, self.observations) / W  # = δ·μ̂_i
                mu_new[i] = m_est / self.delta

                # σ̂_i² = Σ_t (z_t − δ·μ̂_i)² γ_t(i) / (δ · Σ_t γ_t(i))
                var_est = np.dot(w, (self.observations - m_est) ** 2) / W  # = δ·σ̂_i²
                sigma_new[i] = np.sqrt(max(var_est / self.delta, 1e-12))

            # Transition matrix avec prior Dirichlet (MAP) :
            # P̂[i,j] = (n_ij + α_ij - 1) / Σ_k (n_ik + α_ik - 1)
            P_new = np.zeros((d, d))
            for i in range(d):
                for j in range(d):
                    n_ij = xi[:, i, j].sum()
                    P_new[i, j] = n_ij + alpha_prior[i, j] - 1.0

                # Projection sur le simplexe (clamp négatifs à 0, normaliser)
                P_new[i] = np.maximum(P_new[i], 0.0)
                row_sum = P_new[i].sum()
                if row_sum > 0:
                    P_new[i] /= row_sum
                else:
                    P_new[i] = P[i]

            # π̂ = γ_0
            pi_new = gamma[0].copy()

            # ── Convergence ──
            diff = (
                np.max(np.abs(mu_new - mu))
                + np.max(np.abs(sigma_new - sigma))
                + np.max(np.abs(P_new - P))
            )

            mu, sigma, P, pi = mu_new, sigma_new, P_new, pi_new

            if diff < tol and iteration > 5:
                print(f"Convergence atteinte à l'itération {iteration + 1}")
                break

            prev_ll = log_lik

        # ── Stocker les résultats finaux ──
        self.mu = mu
        self.sigma = sigma
        self.P = P
        self.pi = pi  # FIX : pi = γ_0 (pas pi_filt[-1])
        self.filter = self._compute_filtering(
            mu, sigma, P, pi
        )  # FIX : recalculer avec params finaux

        return mu, sigma, P, pi

    def likelihood(self):
        return np.mean(np.log(np.maximum(self.filter["scales"], 1e-300)))

    def simulate(self, T=10, N=1, pi=None, start_date=None, seed=None):
        delta = self.delta if self.delta is not None else 1.0
        T_steps = int(T / delta) + 1
        if pi is None:
            pi = self.pi
        if seed is not None:
            np.random.seed(seed)
        if start_date is None:
            start_date = pd.Timestamp.today()
        else:
            start_date = pd.to_datetime(start_date, dayfirst=True)

        obs = np.zeros((T_steps + 1, N))
        regimes = np.zeros((T_steps + 1, N), dtype=int)

        for i in range(N):
            obs[0, i] = float(self.z0) if self.z0 is not None else 0.0
            regimes[0, i] = np.random.choice(self.d, p=pi)
            for t in range(1, T_steps + 1):
                regimes[t, i] = np.random.choice(self.d, p=self.P[regimes[t - 1, i], :])
                obs[t, i] = np.random.normal(
                    delta * self.mu[regimes[t, i]],
                    np.sqrt(delta) * self.sigma[regimes[t, i]],
                )

        dates = np.array(
            [
                start_date + pd.Timedelta(days=365.25 * delta * t)
                for t in range(T_steps + 1)
            ]
        )
        df_obs = pd.DataFrame(obs, columns=[f"x_simul_{i + 1}" for i in range(N)])
        df_obs.insert(0, "Date", dates)
        df_regimes = pd.DataFrame(
            regimes, columns=[f"regime_{i + 1}" for i in range(N)]
        )
        df_regimes.insert(0, "Date", dates)

        return df_obs, df_regimes

    def summary(self):
        print("Estimation par Maximum de Vraisemblance (Hardy):")
        for i in range(self.d):
            print(f"  Régime {i}: mu = {self.mu[i]:.6f}, sigma = {self.sigma[i]:.6f}")
        print(f"  P =\n{self.P}")
        print(f"  pi = {self.pi}")
        if self.filter is not None:
            print(f"  Log-vraisemblance = {self.likelihood():.4f}")


class HardyMultivariate:
    def __init__(self, params=None):
        if params is None:
            params = [
                np.array([[-10, 10], [-10, 10]]),
                np.array(
                    [np.eye(2) * 10.0, np.eye(2) * 10.0]
                ),  # Sigma = Cholesky/Volatilité, donc 10 au lieu de 100
                np.array([[0.9, 0.1], [0.05, 0.95]]),
                np.array([0.7, 0.3]),
            ]
        self.ajustement_var = None
        self.df_path = None
        self.df = None
        self.df_brut = None
        self.year_col = None
        self.value_cols = None
        self.type_data = None
        self.frequence = None
        self.start_year = None
        self.end_year = None
        self.mu, self.Sigma, self.P, self.pi = params
        self.delta = 1
        self.d = self.mu.shape[0]
        self.p = self.mu.shape[1]
        self.observations = None
        self.filter = None
        self.initial_values = {}

        # --- Nouveaux attributs ajoutés ---
        self.log_lik = None
        self.r_squared = None
        self.r_pseudo_squared = None
        self.sw_test = None
        self.residuals = None

    def _mv_normal_pdf(self, z, mean, cov):
        return stats.multivariate_normal.pdf(z, mean=mean, cov=cov, allow_singular=True)

    def _emission_probabilities(self, z, mu=None, Sigma=None):
        """b_i(z) = φ(z; δ·μ_i, δ·Σ_i)"""
        if mu is None:
            mu = self.mu
        if Sigma is None:
            Sigma = self.Sigma
        b = np.array(
            [
                # La loi normale multivariée attend une matrice de covariance, on la recrée avec Sigma @ Sigma.T
                self._mv_normal_pdf(
                    z, self.delta * mu[i], self.delta * (Sigma[i] @ Sigma[i].T)
                )
                for i in range(self.d)
            ]
        )
        return b

    def _compute_filtering(self, mu=None, Sigma=None, P=None, pi=None):
        if mu is None:
            mu = self.mu
        if Sigma is None:
            Sigma = self.Sigma
        if P is None:
            P = self.P
        if pi is None:
            pi = self.pi

        d = self.d
        n = len(self.observations)
        pi_pred = np.zeros((n, d))
        pi_filt = np.zeros((n, d))
        scales = np.zeros(n)
        B = np.zeros((n, d))

        pi_pred[0] = pi
        B[0] = self._emission_probabilities(self.observations[0], mu, Sigma)
        c0 = np.dot(pi_pred[0], B[0])
        scales[0] = c0 if c0 > 1e-300 else 1e-300
        pi_filt[0] = (pi_pred[0] * B[0]) / scales[0]

        for t in range(1, n):
            pi_pred[t] = P.T @ pi_filt[t - 1]
            B[t] = self._emission_probabilities(self.observations[t], mu, Sigma)
            ct = np.dot(pi_pred[t], B[t])
            scales[t] = ct if ct > 1e-300 else 1e-300
            pi_filt[t] = (pi_pred[t] * B[t]) / scales[t]

        return {"pi_pred": pi_pred, "pi_filt": pi_filt, "scales": scales, "B": B}

    # ─────────────────────────────────────────────────────────────────
    #  FORWARD-BACKWARD (Baum-Welch)
    # ─────────────────────────────────────────────────────────────────
    def _forward_backward(self, mu, Sigma, P, pi):
        filt = self._compute_filtering(mu, Sigma, P, pi)
        alpha = filt["pi_filt"]
        scales = filt["scales"]
        B = filt["B"]

        n, d = alpha.shape

        # Backward
        beta = np.ones((n, d))
        for t in range(n - 2, -1, -1):
            beta[t] = P @ (B[t + 1] * beta[t + 1]) / scales[t + 1]

        # Smoothed marginals
        gamma = alpha * beta
        gamma_sum = gamma.sum(axis=1, keepdims=True)
        gamma_sum = np.maximum(gamma_sum, 1e-300)
        gamma = gamma / gamma_sum

        # Smoothed joints
        xi = np.zeros((n - 1, d, d))
        for t in range(n - 1):
            tmp = (alpha[t][:, None] * P) * (B[t + 1] * beta[t + 1])[None, :]
            tmp /= scales[t + 1]
            s = tmp.sum()
            if s > 0:
                tmp /= s
            xi[t] = tmp

        log_lik = np.sum(np.log(np.maximum(scales, 1e-300)))
        return gamma, xi, log_lik, filt

    # ─────────────────────────────────────────────────────────────────
    #  Initialisation guidée par les données
    # ─────────────────────────────────────────────────────────────────
    def _initialize_from_data(self):
        n, p = self.observations.shape
        d = self.d

        if d == 1:
            mu = (self.observations.mean(axis=0) / self.delta).reshape(1, p)
            cov = np.cov(self.observations, rowvar=False, ddof=0) / self.delta
            # Sigma est la matrice de volatilité, donc la racine de Cholesky
            Sigma = np.linalg.cholesky(cov + np.eye(p) * 1e-6).reshape(1, p, p)
            return mu, Sigma, np.array([[1.0]]), np.array([1.0])

        # Trier par norme et découper en quantiles
        norms = np.linalg.norm(self.observations, axis=1)
        indices = np.argsort(norms)
        group_size = n // d

        mu = np.zeros((d, p))
        Sigma = np.zeros((d, p, p))

        for i in range(d):
            start = i * group_size
            end = (i + 1) * group_size if i < d - 1 else n
            group = self.observations[indices[start:end]]

            mu[i] = group.mean(axis=0) / self.delta
            centered = group - group.mean(axis=0)
            cov = (centered.T @ centered) / max(len(group) - 1, 1)
            # Transformation en matrice de volatilité (Cholesky)
            Sigma[i] = np.linalg.cholesky(cov / self.delta + np.eye(p) * 1e-6)

        pi = np.ones(d) / d
        P = np.full((d, d), 0.05 / max(d - 1, 1))
        np.fill_diagonal(P, 0.95)
        P = P / P.sum(axis=1, keepdims=True)

        return mu, Sigma, P, pi

    # ─────────────────────────────────────────────────────────────────
    #  FIT MODEL
    # ─────────────────────────────────────────────────────────────────
    def fit_model(
        self,
        df_path_ajustement_var: dict[str, list] = None,
        max_iter=200,
        tol=1e-6,
        dirichlet_diag=None,
        dirichlet_offdiag=None,
    ):
        # ─── Chargement des données (inchangé) ──────────────
        observations = {}
        deltas = []
        frequencies = []
        hardy_instances = {}

        for key, (df_path, ajustement_var) in df_path_ajustement_var.items():
            dummy_params = [np.zeros(2), np.ones(2), np.eye(2), np.array([0.5, 0.5])]
            hardy = Hardy(
                params=dummy_params, df_path=df_path, ajustement_var=ajustement_var
            )
            data, delta = hardy._prepare_data()
            deltas.append(delta)
            frequencies.append(hardy.frequence)
            data = data.rename(columns={"log_variation": f"log_variation_{key}"})
            observations[key] = data
            self.initial_values[key] = hardy.z0
            hardy_instances[key] = hardy

        has_mixed_frequencies = len(set(frequencies)) > 1
        if has_mixed_frequencies:
            print("Mixed frequencies detected. Converting all data to yearly format.")
            if "year" in frequencies:
                for key, data in observations.items():
                    if key == "Date":
                        continue
                    if any("-" in str(date) for date in data["Date"]):
                        data["year"] = data["Date"].astype(str).str[-4:].astype(int)
                        yearly_data = data.groupby("year").sum().reset_index()
                        yearly_data["Date"] = yearly_data["year"].astype(str)
                        yearly_data = yearly_data.drop(columns=["year"])
                        observations[key] = yearly_data
                self.delta = 1.0
            else:
                warnings.warn("Mixed frequencies but no yearly data.")
        else:
            if not all(x == deltas[0] for x in deltas):
                warnings.warn("All delta times are not equal.")
            self.delta = np.mean(deltas)

        dfs = [
            df.astype({"Date": str}).set_index("Date") for df in observations.values()
        ]
        if len(dfs) > 1:
            self.df = pd.concat(dfs, axis=1, join="inner").reset_index()
        elif len(dfs) == 1:
            self.df = dfs[0].reset_index()

        first_key = list(hardy_instances.keys())[0]
        self.df_brut = hardy_instances[first_key].df_brut
        self.year_col = hardy_instances[first_key].year_col
        self.value_col = hardy_instances[first_key].value_col
        self.type_data = hardy_instances[first_key].type_data
        self.observations = self.df.drop(columns=["Date"]).to_numpy()

        n = len(self.observations)
        d = self.d
        p = self.p

        # ── Initialisation ──
        mu, Sigma, P, pi = self._initialize_from_data()

        # ── Prior Dirichlet ──
        if dirichlet_diag is None:
            dirichlet_diag = 25.0
        if dirichlet_offdiag is None:
            dirichlet_offdiag = 2

        alpha_prior = np.full((d, d), dirichlet_offdiag)
        np.fill_diagonal(alpha_prior, dirichlet_diag)

        # ── Boucle EM ──
        for iteration in range(max_iter):

            gamma, xi, log_lik, filt = self._forward_backward(mu, Sigma, P, pi)

            # === M-step ===
            mu_new = np.zeros((d, p))
            Sigma_new = np.zeros((d, p, p))

            for i in range(d):
                w = gamma[:, i]
                W = w.sum()

                if W < 1e-10:
                    mu_new[i] = mu[i]
                    Sigma_new[i] = Sigma[i]
                    continue

                # μ̂_i
                m_est = (w[:, None] * self.observations).sum(axis=0) / W  # = δ·μ̂_i
                mu_new[i] = m_est / self.delta

                # Σ̂_i
                X_centered = self.observations - m_est  # z_t − δ·μ̂_i
                cov_est = (
                    w[:, None, None] * np.einsum("tn,tp->tnp", X_centered, X_centered)
                ).sum(
                    axis=0
                ) / W  # = δ·Σ̂_i (Ceci est la variance)

                # Sigma = Matrice de volatilité (Cholesky)
                Sigma_new[i] = np.linalg.cholesky(
                    cov_est / self.delta + np.eye(p) * 1e-6
                )

            # P̂ avec prior Dirichlet
            P_new = np.zeros((d, d))
            for i in range(d):
                for j in range(d):
                    P_new[i, j] = xi[:, i, j].sum() + alpha_prior[i, j] - 1.0
                P_new[i] = np.maximum(P_new[i], 0.0)
                row_sum = P_new[i].sum()
                if row_sum > 0:
                    P_new[i] /= row_sum
                else:
                    P_new[i] = P[i]

            pi_new = gamma[0].copy()

            # ── Convergence ──
            diff = (
                np.max(np.abs(mu_new - mu))
                + np.max(np.abs(Sigma_new - Sigma))
                + np.max(np.abs(P_new - P))
            )

            mu, Sigma, P, pi = mu_new, Sigma_new, P_new, pi_new

            if diff < tol and iteration > 5:
                print(f"Convergence atteinte à l'itération {iteration + 1}")
                break

        # ── Résultats finaux ──
        self.mu = mu
        self.Sigma = Sigma
        self.P = P
        self.pi = pi
        self.compute_residuals()
        # --------------

        return mu, Sigma, P, pi

    # ─────────────────────────────────────────────────────────────────
    #  RESIDUALS
    # ─────────────────────────────────────────────────────────────────
    def compute_residuals(self):
        self.filter = self._compute_filtering(self.mu, self.Sigma, self.P, self.pi)

        # --- Ajouts ---
        self.log_lik = self.likelihood()
        try:
            # 1. Calcul de la prédiction conditionnelle à chaque instant t
            # On pondère les moyennes (mu) selon les probabilités prédites (pi_pred) des régimes
            pi_pred = self.filter["pi_pred"]
            z_pred = pi_pred @ (self.delta * self.mu)

            # 2. Résidus multivariés (observations réelles - prédictions)
            residuals = self.observations - z_pred

            # 3. Calcul du R² (r2_score gère les tableaux 2D nativement en faisant une moyenne)
            self.r_squared = r2_score(self.observations, z_pred)

            # 4. Calcul du pseudo R² par rapport à un modèle nul (une seule loi normale multivariée)
            mean_null = np.mean(self.observations, axis=0)
            cov_null = np.cov(self.observations, rowvar=False)
            ll_null = np.sum(
                stats.multivariate_normal.logpdf(
                    self.observations, mean=mean_null, cov=cov_null, allow_singular=True
                )
            )

            if ll_null != 0:
                self.r_pseudo_squared = 1 - (self.log_lik / ll_null)
            else:
                self.r_pseudo_squared = None

            # 5. Test de Shapiro-Wilk (sur les résidus standardisés globalement)
            res_flat = residuals.flatten()
            std_res = np.std(res_flat)
            if std_res > 0:
                self.sw_test = stats.shapiro(res_flat / std_res)
            else:
                self.sw_test = None

            # 6. Création du DataFrame pour stocker les résidus
            if self.df is not None and "Date" in self.df.columns:
                dates = self.df["Date"].values
            else:
                dates = np.arange(len(self.observations))

            self.residuals = pd.DataFrame(
                residuals, columns=[f"Residuals_{i + 1}" for i in range(self.p)]
            )
            self.residuals.insert(0, "Date", dates)

        except Exception as e:
            print(f"Erreur lors du calcul des résidus : {e}")
            return None

    def likelihood(self):
        return np.mean(np.log(np.maximum(self.filter["scales"], 1e-300)))

    def simulate(self, T=10, N=1, pi=None, start_date=None, seed=None, use_steps=False):
        delta = self.delta
        T_steps = int(T / delta)
        if pi is None:
            pi = self.pi
        if seed is not None:
            np.random.seed(seed)

        obs = np.zeros((T_steps + 1, N, self.p))
        regimes = np.zeros((T_steps + 1, N), dtype=int)

        for i in range(N):
            if self.initial_values:
                obs[0, i] = np.array(
                    [
                        float(self.initial_values.get(str(k + 1), 0.0))
                        for k in range(self.p)
                    ],
                    dtype=float,
                )
            else:
                obs[0, i] = np.zeros(self.p)
            regimes[0, i] = np.random.choice(self.d, p=pi)
            for t in range(1, T_steps + 1):
                regimes[t, i] = np.random.choice(self.d, p=self.P[regimes[t - 1, i], :])

                # Recomposition de la variance (Sigma @ Sigma.T) pour le générateur aléatoire
                obs[t, i] = np.random.multivariate_normal(
                    delta * self.mu[regimes[t, i]],
                    delta * (self.Sigma[regimes[t, i]] @ self.Sigma[regimes[t, i]].T),
                )
                # obs[t, i] = obs[t - 1, i] + increment

        if use_steps or start_date is None:
            time_axis = np.arange(T_steps + 1)
            time_col = "step"
        else:
            start_date = pd.to_datetime(start_date, dayfirst=True)
            time_axis = np.array(
                [
                    start_date + pd.Timedelta(days=365.25 * delta * t)
                    for t in range(T_steps + 1)
                ]
            )
            time_col = "Date"

        obs_flat = obs.reshape(T_steps + 1, N * self.p)
        df_obs = pd.DataFrame(
            obs_flat,
            columns=[
                f"x{k + 1}_simul_{i + 1}" for i in range(N) for k in range(self.p)
            ],
        )
        df_obs.insert(0, time_col, time_axis)
        df_regimes = pd.DataFrame(
            regimes, columns=[f"regime_{i + 1}" for i in range(N)]
        )
        df_regimes.insert(0, time_col, time_axis)

        factor_dfs = {
            f"{k + 1}": df_obs[
                [time_col] + [f"x{k + 1}_simul_{i + 1}" for i in range(N)]
            ].copy()
            for k in range(self.p)
        }
        return df_obs, df_regimes, factor_dfs

    def get_initial_values(self):
        return self.initial_values

    def summary(self):
        self.compute_residuals()
        print("Estimation par Maximum de Vraisemblance (multivariée):")
        for i in range(self.d):
            print(f"\n  Régime {i}:")
            print(f"    mu        = {self.mu[i]}")
            print(f"    Sigma (volatilité) =\n{self.Sigma[i]}")
        print(f"\n  P =\n{self.P}")
        print(f"  pi = {self.pi}")
        if self.filter is not None:
            print(f"  Log-vraisemblance = {self.likelihood():.4f}")

    # --- Nouvelle méthode ajoutée ---
    def get_results(self):
        """
        Retourne un dictionnaire contenant les paramètres estimés et les métriques d'ajustement.
        """
        self.compute_residuals()
        return {
            "mu": self.mu,
            "Sigma": self.Sigma,
            "P": self.P,
            "pi": self.pi,
            "r_squared": self.r_squared,
            "r_pseudo_squared": self.r_pseudo_squared,
            "log_lik": self.log_lik,
            "Shapiro-Wilk": self.sw_test,
        }
