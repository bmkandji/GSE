from typing import Optional
from typing import List
import numpy
import statsmodels.api as sm
from scipy.optimize import minimize, minimize_scalar
from gse_engine.ahlgrim.tools import *
from scipy import stats
from gse_engine.db import read_sql_sheet, write_sql_sheet


class Ornstein_Uhlenbeck:
    def __init__(self, params=None, fix_kappa=None):
        """
        Initialise la régression AR(1) sur les données.
        """
        self._type = "OU"
        self.df_path = {"path": None, "index": "Index"}
        self.ajustement_var = None
        # Init du scénario climatique
        self.df_sc = None
        self.date_col_sc = None
        self.mu_col_sc = None
        self.filtres_sc = None
        # Init de la donnée historique
        self.year_col = None
        self.value_col = None
        self.type_data = None
        self.start_year = None
        self.end_year = None
        self.df = None
        self.df_brut = None

        self.model = None
        self.alpha = None
        self.phi = None
        self.sigma_epsilon = None
        self.residuals = None  # Stocke les résidus
        self.sw_test = None
        self.correlation_epsilon_rl = None
        self.log_lik = None  # log-vraisemblance
        self.r_squared = None
        self.r_pseudo_squared = None
        self.z0 = None

        if params is None:
            params = [None] * 3

        self.kappa, self.mu, self.sigma = params
        self.fix_kappa = fix_kappa
        self.rho = None
        self.z_t = None
        self.z_tm1 = None
        # Scenario means: aligned with (z_tm1, z_t) pairs; None on pre-scenario period
        self.mu_scenario = None  # length = len(z_t), value or NaN
        self.warnings = []  # List to store calibration warnings

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

            if len(self.year_col) == 0 or len(self.value_col) == 0:
                raise ValueError(
                    "Les colonnes pour 'Date' , 'type_data' et 'Variable' n'ont pas été trouvées dans le fichier Index."
                )

            self.df_brut = read_sql_sheet(
                self.df_path["data"], self.df_path["path"], dtype={self.year_col: str}
            )

            # Sort values by year_col to ensure correct order
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

                    df_log_var = adjust_dataframes(
                        df_x=df_log_var,
                        df_z=ajustement_df,
                        df_x_date_col=self.year_col,
                        df_z_date_col=self.ajustement_var.year_col,
                        df_x_var_col="log_variation",
                        df_z_var_col="log_variation",
                    )
                except Exception as e:
                    print(f"Erreur sur la variable d'ajustement (ex: inflation) : {e}")
            # Tronquer les données après start_year (end_year est INCLUSIF)
            self.df = df_log_var[
                (df_log_var[self.year_col] >= self.start_year)
                & (df_log_var[self.year_col] <= self.end_year)
            ][[self.year_col, "log_variation"]]
            log_variation_series = self.df.set_index(self.year_col)["log_variation"]
            print("self.df after preparation", self.df)
            # Création des séries temporelles décalées
            self.z_t = log_variation_series[1:].reset_index(drop=True)  # z_t
            self.z_tm1 = log_variation_series[:-1].reset_index(drop=True)  # z_{t-1}
            self.z0 = self.df["log_variation"].iloc[-1]

        except Exception as e:
            print(f"Erreur lors de la préparation des données : {e}")

    def _neg_log_likelihood(self, params):
        """
        Calcule la négative log-vraisemblance pour le modèle OU/ABM généralisé.

        Discrétisation Ψ-basée (valide pour κ > 0 et κ = 0) :
            z_t = e^{−κδ} z_{t-1} + Ψ(κ,δ)·γ·μ + σ·√Ψ(2κ,δ)·ε

        params : array de 3 éléments [mu, kappa, sigma]
        """

        mu, kappa, sigma = params.tolist()
        delta = 1.0  # pas de calibration annuel

        # ── Variance conditionnelle : σ² · Ψ(2κ, δ) ──
        sigma_epsilon = sigma * np.sqrt(Psi(2.0 * kappa, delta))

        if sigma_epsilon <= 0:
            return 1e12  # garde contre les cas dégénérés

        # ── Coefficients de transition ──
        exp_z = ar_coeff(kappa, delta)  # 0 pour ABM (données i.i.d.)
        gam = gamma_coeff(kappa)  # γ = 𝟙{κ=0} + κ
        psi_kd = Psi(kappa, delta)  # Ψ(κ, δ)

        # Build mu vector: constant mu on pre-scenario, scenario mu on post-scenario
        if self.mu_scenario is not None:
            mu_vec = self.mu_scenario["mu_scenario"].copy()
            mu_vec[np.isnan(mu_vec)] = mu  # pre-scenario: use estimated mu
        else:
            mu_vec = np.full(len(self.z_tm1), mu)

        # ── Moyenne conditionnelle : e^{−κδ} z_{t-1} + Ψ(κ,δ)·γ·μ ──
        x_pred = exp_z * self.z_tm1 + psi_kd * gam * mu_vec

        # Calcul de l'erreur pour chaque observation
        errors = self.z_t - x_pred

        # Pour chaque vecteur d'erreur e, calcul de e.T @ inv_cov @ e
        likelihood_terms = np.mean(
            np.array([e**2 / sigma_epsilon**2 for e in errors])
        ) + np.log(sigma_epsilon**2)

        return likelihood_terms

    def fit_model(
        self,
        df_path,
        ajustement_var: Optional["Ornstein_Uhlenbeck"] = None,
        df_mu_path: dict = None,
    ):
        """
        Estime directement par vraisemblance les paramètres du modèle :
          - kappa: κ^(z)
          - sigma: σ^(z)
          - mu: μ^(z) (estimé uniquement sur la période pré-scénario)

        Si df_mu_path est fourni, les mu des scénarios sont utilisés sur la période
        post-scénario pour calibrer kappa et sigma. Le mu estimé ne porte alors que
        sur la période pré-scénario (Section 4.1 du papier).

        Paramètre fix_kappa :
          - None (défaut) : κ est estimé librement (κ ≥ 0)
          - 0.0           : force le modèle ABM (κ = 0), seuls μ et σ sont estimés
          - toute valeur  : fixe κ à cette valeur, seuls μ et σ sont estimés
        """
        fix_kappa = self.fix_kappa
        self.df_path = df_path
        self.ajustement_var = ajustement_var
        self._prepare_data()
        self._prepare_mu_scenario(df_mu_path)
        try:
            if fix_kappa is not None:
                # ── κ fixé : optimiser seulement (μ, σ) ──
                fixed_k = float(fix_kappa)

                def _nll_reduced(params_2d):
                    """Neg-log-lik avec κ fixé, params_2d = [mu, sigma]."""
                    full = np.array([params_2d[0], fixed_k, params_2d[1]])
                    return self._neg_log_likelihood(full)

                initial_guess = np.array([2.5, 2.0])  # [mu, sigma]
                bounds = [(None, None), (1e-10, None)]

                result = minimize(
                    _nll_reduced,
                    initial_guess,
                    method="L-BFGS-B",
                    bounds=bounds,
                )
                if not result.success:
                    print("L'optimisation ML (κ fixé) n'a pas convergé.")

                self.mu, self.sigma = result.x
                self.kappa = fixed_k
            else:
                # ── κ libre : optimiser (μ, κ, σ) ──
                initial_guess = np.array([2.5, 1.0, 2.0])  # [mu, kappa, sigma]
                bounds = [(None, None), (0.0, None), (1e-10, None)]

                result = minimize(
                    self._neg_log_likelihood,
                    initial_guess,
                    method="L-BFGS-B",
                    bounds=bounds,
                )
                if not result.success:
                    print("L'optimisation ML n'a pas convergé.")

                self.mu, self.kappa, self.sigma = result.x

            # ── Post-traitement commun ──
            # σ̃ = σ · √Ψ(2κ, δ)  (écart-type, cohérent avec fit_model_ols)
            self.compute_residuals()

        except Exception as e:
            print(f"Erreur lors de l'estimation ML : {e}")

    def _prepare_mu_scenario(self, df_mu_path: dict = None):
        """
        Charge les mu de scénarios et construit un vecteur aligné sur les paires
        (z_{t-1}, z_t). Pour chaque observation t, si la date t-1 tombe dans la
        période scénario, on utilise le mu du scénario ; sinon NaN (= mu constant estimé).

        Le vecteur résultant self.mu_scenario a la même longueur que self.z_t.
        """
        self.mu_scenario = None
        if df_mu_path is None:
            return

        try:
            dico = read_sql_sheet(df_mu_path["index"], df_mu_path["path"])
            if "Refrence" not in dico.columns or "data" not in dico.columns:
                return

            self.date_col_sc = dico.loc[dico["Refrence"] == "Date", "data"].values[
                0
            ]  # Je récupère la colonne date du scénario climatique
            self.mu_col_sc = dico.loc[dico["Refrence"] == "Variable", "data"].values[
                0
            ]  # Je récupère la colonne avec le mu du scénario climatique
            mu_adjusted_values = dico.loc[
                dico["Refrence"] == "mu_adjusted", "data"
            ].values
            mu_adjusted_raw = (
                mu_adjusted_values[0] if len(mu_adjusted_values) > 0 else False
            )
            self.type_data_sc = dico.loc[dico["Refrence"] == "Type", "data"].values[0]
            if isinstance(mu_adjusted_raw, str):
                mu_adjusted = mu_adjusted_raw.strip().lower() in (
                    "true",
                    "1",
                    "yes",
                    "y",
                    "oui",
                )
            else:
                mu_adjusted = bool(mu_adjusted_raw)
            DataLeaf = dico.loc[dico["Refrence"] == "DataLeaf", "data"].values[0]
            self.ToPass_sc = dico.loc[dico["Refrence"] == "ToPass", "data"].values[0]

            self.df_sc = read_sql_sheet(
                DataLeaf, df_mu_path["path"], dtype={self.date_col_sc: str}
            )

            if self.ajustement_var is not None and not mu_adjusted:
                # Si on a utilisé une variable d'ajustement pour le facteur de risque

                try:
                    # Copier et renommer la colonne de date du scénario climatique de l'ajustement pour correspondre à celle du modèle actuel
                    ajustement_df = (
                        self.ajustement_var.df_sc.copy()
                    )  # Récupère le scénario climatique de la variable d'ajustement
                    self.df_sc = adjust_dataframes(
                        df_x=self.df_sc,
                        df_z=ajustement_df,
                        df_x_date_col=self.date_col_sc,
                        df_z_date_col=self.ajustement_var.date_col_sc,
                        df_x_var_col=self.mu_col_sc,
                        df_z_var_col=self.ajustement_var.mu_col_sc,
                    )
                except Exception as e:
                    print(f"Erreur sur la variable d'ajustement (ex: inflation) : {e}")

            if self.ToPass_sc == "false":
                val_to_filtre = dico.loc[dico["Refrence"] == "ToFiltre", "data"].values[
                    0
                ]
                ToFiltre = val_to_filtre.split("-") if pd.notna(val_to_filtre) else None
                val_value_filtre = dico.loc[
                    dico["Refrence"] == "ValueFiltre", "data"
                ].values[0]
                ValueFiltre = (
                    val_value_filtre.split("-") if pd.notna(val_value_filtre) else None
                )
                self.filtres_sc = (
                    [[tf, vf] for tf, vf in zip(ToFiltre, ValueFiltre)]
                    if (ToFiltre is not None and ValueFiltre is not None)
                    else [[None, None]]
                )
                from gse_engine.ahlgrim.tools import filtrer_et_interpoler

                df_sc_fil_int = filtrer_et_interpoler(
                    self.df_sc,
                    colonne_date=self.date_col_sc,
                    colonne_valeur=self.mu_col_sc,
                    filtres=self.filtres_sc,
                    start_date=int(self.df_sc[self.date_col_sc][0]),
                    mu0=self.df_sc[self.mu_col_sc][0],
                )
                df_sc_fil_int[self.date_col_sc] = df_sc_fil_int[
                    self.date_col_sc
                ].astype(int)
            else:
                df_sc_fil_int = self.df_sc.copy()

            df_sc_log = log_variation(
                df_sc_fil_int,
                self.mu_col_sc,
                self.date_col_sc,
                name=self.mu_col_sc,
                type_data=self.type_data_sc,
            )

            # Build a lookup: year -> scenario mu
            sc_lookup = dict(
                zip(
                    df_sc_log[self.date_col_sc].astype(int),
                    df_sc_log[self.mu_col_sc].astype(float),
                )
            )

            # Build the mu_scenario_vec aligned with z_tm1 (the "previous" dates)
            # self.df has columns [year_col, "log_variation"] and is already filtered
            dates_tm1 = self.df[self.year_col].iloc[1:].reset_index(drop=True)
            n = len(dates_tm1)
            mu_vec = np.full(n, np.nan)
            for i in range(n):
                yr = int(dates_tm1.iloc[i])
                if yr in sc_lookup:
                    mu_vec[i] = sc_lookup[yr]

            # Only set if at least some scenario values exist
            if not np.all(np.isnan(mu_vec)):
                self.mu_scenario = pd.DataFrame(
                    {"Date": dates_tm1, "mu_scenario": mu_vec}
                )
                print(
                    f"  [OU] Scenario mu loaded: {np.sum(~np.isnan(mu_vec))}/{n} obs with scenario mu"
                )
            else:
                # Get scenario date range
                scenario_start = (
                    int(df_sc_fil_int[self.date_col_sc].min())
                    if len(df_sc_fil_int) > 0
                    else None
                )
                scenario_end = (
                    int(df_sc_fil_int[self.date_col_sc].max())
                    if len(df_sc_fil_int) > 0
                    else None
                )
                calib_start = int(dates_tm1.iloc[0]) if len(dates_tm1) > 0 else None
                calib_end = int(dates_tm1.iloc[-1]) if len(dates_tm1) > 0 else None

                warning_msg = "<strong>No observation from the chosen scenario matches the observation dates:</strong><ul>"
                if scenario_start and scenario_end:
                    warning_msg += (
                        f"<li>Scenario period: {scenario_start} - {scenario_end}</li>"
                    )
                if calib_start and calib_end:
                    warning_msg += (
                        f"<li>Calibration period: {calib_start} - {calib_end}</li>"
                    )
                warning_msg += "</ul>"

                self.warnings.append(warning_msg)
                print("  [OU] No scenario mu matched observation dates")

        except Exception as e:
            print(f"  [OU] Erreur chargement mu scenario: {e}")

    def compute_residuals(self):
        """
        Calcule les résidus du modèle OU/ABM généralisé :
            ε_n = z_t − [e^{−κδ} z_{t-1} + Ψ(κ,δ)·γ·μ]
        """

        self.sigma_epsilon = self.sigma * np.sqrt(Psi(2.0 * self.kappa, 1.0))

        self.log_lik = -0.5 * (
            self._neg_log_likelihood(params=np.array([self.mu, self.kappa, self.sigma]))
            + np.log(2 * np.pi)
        )
        try:
            delta = 1.0  # pas de calibration annuel

            # Calcul des coefficients
            exp_z = ar_coeff(self.kappa, delta)  # 0 pour ABM (données i.i.d.)
            gam = gamma_coeff(self.kappa)
            psi_kd = Psi(self.kappa, delta)

            # Build mu vector (same logic as _neg_log_likelihood)
            if self.mu_scenario is not None:
                mu_vec = self.mu_scenario["mu_scenario"].copy()
                mu_vec[np.isnan(mu_vec)] = self.mu
            else:
                mu_vec = np.full(len(self.z_tm1), self.mu)

            # Prédiction : e^{−κδ} z_{t-1} + Ψ(κ,δ)·γ·μ
            r_pred = exp_z * self.z_tm1 + psi_kd * gam * mu_vec

            df_tronc = self.df[
                (self.df[self.year_col] >= self.start_year)
                & (self.df[self.year_col] <= self.end_year)
            ]  # Appliquer le filtre (end_year est INCLUSIF)
            dates = (
                df_tronc[self.year_col].iloc[1:].reset_index(drop=True)
            )  # Aligner avec r_t
            # Calcul des résidus
            residuals = self.z_t - r_pred

            # Écart-type conditionnel : σ · √Ψ(2κ, δ)
            sigma_cond = np.sqrt(self.sigma**2 * Psi(2.0 * self.kappa, delta))
            if sigma_cond > 0:
                self.sw_test = stats.shapiro(residuals / sigma_cond)
            self.r_squared = r2_score(self.z_t, r_pred)
            self.r_pseudo_squared = 1 - (self.log_lik / log_likelihood_null(self.z_t))
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
        z0,
        start_date,
        df_mu_path: dict = None,
        delta=1,
        T=None,
        N=1,
        seed=None,
        deltas_noise_in: List[np.ndarray] = None,
    ):
        """
        Simule N trajectoires du processus d'Ornstein-Uhlenbeck discrétisé à partir de z0,
        avec des dates exprimées en années. start_date (par exemple 2020) doit toujours être fourni.

        Cas 1 : Si df_mu_path est fourni
          - On charge un fichier Excel qui contient une colonne de dates (en années) et de μ.
        Cas 2 : Si df_mu_path n'est pas fourni
          - On simule avec un pas constant self.delta (en années) et une valeur constante self.mu.

        Paramètres :
          - z0         : condition initiale (valeur unique).
          - df_mu_path : chemin vers le fichier Excel (optionnel).
          - T          : nombre d'étapes à simuler (pour le cas df_mu_path=None, T doit être fourni).
          - N          : nombre de trajectoires à simuler.
          - seed       : graine aléatoire pour la reproductibilité (optionnel).
          - start_date : année de départ (ex : 2020) (obligatoire).

        Retourne :
          Un DataFrame avec T+1 lignes, incluant :
             - une colonne 'Date' contenant l'année de simulation,
             - une colonne 'mu' (avec la valeur initiale z0 en ligne 0, puis μ pour les itérations),
             - une colonne par trajectoire simulée (z_simul_1, z_simul_2, …).
        """
        if deltas_noise_in is None:
            deltas_noise_in = [None] * 2
        deltas, noise_in = deltas_noise_in
        # Conversion de start_date en datetime
        start_date = pd.to_datetime(start_date)
        if seed is not None:
            np.random.seed(seed)

        # Cas 1 : Utilisation de df_mu_path
        if df_mu_path is not None:

            # If df_sc was never loaded (e.g. historically-calibrated model with
            # df_mu_path injected in uncertainty mode), initialise it now.
            if self.df_sc is None:
                self._prepare_mu_scenario(df_mu_path)

            # df = read_sql_sheet(DataLeaf, df_mu_path["path"], dtype={date_col: str})
            df = self.df_sc.copy()
            if self.filtres_sc is not None:
                df = filtrer_et_interpoler(
                    df,
                    colonne_date=self.date_col_sc,
                    colonne_valeur=self.mu_col_sc,
                    filtres=self.filtres_sc,
                    start_date=int(self.df_sc[self.date_col_sc][0]),
                )
                df[self.date_col_sc] = df[self.date_col_sc].astype(int)

                df = log_variation(
                    df,
                    self.mu_col_sc,
                    self.date_col_sc,
                    self.mu_col_sc,
                    self.type_data_sc,
                )

                df = df[df[self.date_col_sc] >= start_date.year]

            # ─────────────────────────────────────────────────────────────────────────
            df[self.date_col_sc] = pd.to_datetime(df[self.date_col_sc], format="%Y")
            df.sort_values(by=self.date_col_sc, inplace=True)
            df.reset_index(drop=True, inplace=True)

            # Déterminer T (après annualisation, on a exactement T+1 lignes)
            if T is None:
                T = len(df) - 1
            else:
                T = min(T, len(df) - 1)

            # Truncate df to only have T+1 rows (from index 0 to T)
            df = df.iloc[: T + 1].copy()

            # Initialisation du tableau des trajectoires : (T+1) points pour N trajectoires
            z = np.zeros((T + 1, N))
            z[0, :] = z0  # même z0 pour toutes les trajectoires

            if noise_in is None:
                # Calcul des deltas en années pour chaque intervalle de temps
                # Pour i=0, delta correspond à la différence entre df.loc[0, date_col] et start_date
                deltas = [
                    (df.loc[i + 1, self.date_col_sc] - df.loc[i, self.date_col_sc]).days
                    / 365.25
                    for i in range(T)
                ]
                # deltas.insert(0, (df.loc[0, self.date_col_sc] - start_date).days / 365.25)

            # Générer une matrice d'aléas pour chaque étape et chaque trajectoire
            # Ici, on a T étapes (pour i de 1 à T) et N trajectoires
            if noise_in is not None and noise_in.shape != (T, N):
                raise ValueError(f"Dimension {noise_in.shape} != attendues ({T}, {N})")
            epsilon = np.random.normal(0, 1, (T, N)) if noise_in is None else noise_in
            # Boucle sur les étapes temporelles (formule unifiée OU/ABM)
            gam = gamma_coeff(self.kappa)
            for i in range(1, T + 1):
                d = deltas[i - 1]
                # ar_coeff = 0 pour ABM → pas d'accumulation (données i.i.d.)
                exp_term = ar_coeff(self.kappa, d)
                sigma_epsilon = self.sigma * np.sqrt(Psi(2.0 * self.kappa, d))
                # mu_prev est obtenu à partir de df : pour l'étape i, on prend mu de la ligne i
                mu_prev = df.loc[i, self.mu_col_sc]
                # Calcul vectorisé : z = e^{-κδ}·z + Ψ(κ,δ)·γ·μ + σ·√Ψ(2κ,δ)·ε
                z[i, :] = (
                    exp_term * z[i - 1, :]
                    + Psi(self.kappa, d) * gam * mu_prev
                    + sigma_epsilon * epsilon[i - 1, :]
                )

            # Création du vecteur des dates : la première date est start_date, puis les T premières dates de df
            # dates = [start_date] + list(df.loc[:T - 1, date_col])
            dates = list(df.loc[:T, self.date_col_sc])

            # Construction du vecteur des mu simulés :
            # On souhaite avoir en ligne 0 : z0 (comme valeur initiale) et pour i>=1, le mu utilisé (mu_{i-1})
            # mu_sim = np.empty(T + 1)
            # mu_sim[0] = z0  # la condition initiale
            # mu_sim[1:] = df.loc[:T - 1, mu_col].values
            mu_sim = df.loc[:T, self.mu_col_sc].values

        # Cas 2 : df_mu_path n'est pas fourni
        else:
            if T is None:
                raise ValueError(
                    "T doit être fourni lorsque df_mu_path n'est pas donné."
                )
            # On simule avec un pas constant self.delta (exprimé en années) et self.mu constant.
            z = np.zeros((T + 1, N))
            z[0, :] = z0

            # Générer les dates en ajoutant self.delta (en années) à start_date
            dates = [start_date + pd.DateOffset(years=delta * i) for i in range(T + 1)]
            # Construire le vecteur mu
            mu_sim = np.full(T + 1, self.mu)
            if noise_in is None:
                deltas = np.full(T, delta)
            # Générer une matrice d'aléas pour chaque étape et chaque trajectoire
            # Ici, on a T étapes (pour i de 1 à T) et N trajectoires
            if noise_in is not None and noise_in.shape != (T, N):
                raise ValueError(f"Dimension {noise_in.shape} != attendues ({T}, {N})")
            epsilon = np.random.normal(0, 1, (T, N)) if noise_in is None else noise_in
            gam = gamma_coeff(self.kappa)
            for i in range(1, T + 1):
                d = deltas[i - 1]
                # ar_coeff = 0 pour ABM → pas d'accumulation (données i.i.d.)
                exp_term = ar_coeff(self.kappa, d)
                sigma_epsilon = self.sigma * np.sqrt(Psi(2.0 * self.kappa, d))
                z[i, :] = (
                    exp_term * z[i - 1, :]
                    + Psi(self.kappa, d) * gam * self.mu
                    + sigma_epsilon * epsilon[i - 1, :]
                )

        # Construction du DataFrame final : on inclut une colonne 'Date' (année), 'mu'
        # Créer le DataFrame initial
        df_sim = pd.DataFrame({"Date": dates, "mu": mu_sim})

        # Construire un dictionnaire pour les nouvelles colonnes
        new_columns = {f"z_simul_{p + 1}": z[:, p] for p in range(N)}

        # Créer un DataFrame pour ces colonnes avec le même index que df_sim
        df_new = pd.DataFrame(new_columns, index=df_sim.index)

        # Concaténer les DataFrames en une seule opération
        df_sim = pd.concat([df_sim, df_new], axis=1)

        return df_sim, epsilon, deltas

    def summary(self):
        """
        Affiche un résumé des paramètres estimés.
        """
        self.compute_residuals()
        process_type = (
            "ABM (κ=0)" if self.kappa is not None and self.kappa < 1e-10 else "OU"
        )
        print(f"\nType de processus : {process_type}")
        print(f"R² : {self.r_squared:.4f}")
        print(f"pvalue Shapiro-Wilk : {self.sw_test.pvalue:.4f}")
        print("Estimation par Maximum de Vraisemblance:")
        print(f"sigma_epsilon : {self.sigma_epsilon:.4f}")
        print(f"Mu : {self.mu:.4f}")
        print(f"sigma : {self.sigma:.4f}")
        print(f"Kappa : {self.kappa:.6f}")
        print(f"Log-vraisemblance : {self.log_lik:.4f}")

    def get_results(self):
        """
        Retourne un dictionnaire contenant les paramètres estimés.
        """
        self.compute_residuals()
        return {
            "sigma_epsilon": self.sigma_epsilon,
            "mu": self.mu,
            "kappa": self.kappa,
            "sigma": self.sigma,
            "r_squared": self.r_squared,
            "r_pseudo_squared": self.r_pseudo_squared,
            "log_lik": self.log_lik,
            "Shapiro-Wilk": self.sw_test,
        }
