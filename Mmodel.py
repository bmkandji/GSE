from scipy.linalg import block_diag
from .correlations import *
from .hardy import Hardy, HardyMultivariate
from .ornstein_ulhenbeck import Ornstein_Uhlenbeck
from .philips_curve import Phillips_curve
from .two_factor_vasicek import Two_factor_Vasicek
from typing import Union, Dict
import numpy as np
import copy


def compute_D_sigma(factors: dict = None):
    if not factors:
        return np.array([[]])
    keys = list(factors)
    sigmas = [factors[k]["value"].sigma for k in keys]
    D = np.diag(sigmas)

    for i, ki in enumerate(keys):
        vi = factors[ki]["value"]
        for j in range(i + 1, len(keys)):
            vj = factors[keys[j]]["value"]
            if vi._type == "PC" and vj._type == "OU" and vi.inflate == vj:
                D[i, j] = vi.alpha * vj.sigma
    return D


def compute_B(factors: dict = None):
    if not factors:
        return np.array([[]])
    keys = list(factors)
    kappas = [factors[k]["value"].kappa for k in keys]
    B = -np.diag(kappas)
    for i, ki in enumerate(keys):
        vi = factors[ki]["value"]
        for j in range(i + 1, len(keys)):
            vj = factors[keys[j]]["value"]
            if vi._type == "Va2" and vj._type == "OU" and vi.taux_long == vj:
                B[i, j] = vi.kappa
            if vi._type == "PC" and vj._type == "OU" and vi.inflate == vj:
                B[i, j] = -vi.alpha * vj.kappa
    return B


def compute_P(B: np.ndarray = np.array([[]])):
    """
    Décomposition B = P D P⁻¹ pour une matrice triangulaire supérieure stricte
    (zéros exacts sous la diagonale et valeurs propres toutes distinctes).

    • Aucun appel à np.linalg.inv
    • Aucune approximation : toute entrée sous-diagonale ≠ 0 déclenche une erreur.
    """
    B = np.asarray(B)
    if B.size == 0:
        return np.array([[]]), np.array([[]])

    if B.ndim != 2 or B.shape[0] != B.shape[1]:
        raise ValueError("Matrice non carrée")

    # Triangularité supérieure stricte
    if np.any(np.tril(B, k=-1) != 0):
        raise ValueError("B n’est pas strictement triangulaire supérieure")

    n = B.shape[0]
    lam = np.diag(B).copy()

    # ---------- 1) Construction de N (strictement sup.) ----------
    N = np.zeros_like(B)
    for d in range(1, n):  # décalage j-i
        i = np.arange(n - d)  # lignes concernées
        j = i + d  # colonnes associées

        if d == 1:
            S = np.zeros_like(i, dtype=B.dtype)  # vecteur longueur n-d
        else:
            k = np.arange(1, d)  # pas intermédiaires
            # S[i] = Σ_{k=1}^{d-1} B_{i,i+k} * N_{i+k,j}
            S = np.sum(
                B[i[:, None], i[:, None] + k] * N[i[:, None] + k, j[:, None]], axis=1
            )

        denom = lam[i] - lam[j]
        if np.any(denom == 0):
            raise ValueError("Valeurs propres doubles : matrice non diagonalisable")

        N[i, j] = -(B[i, j] + S) / denom

    P = np.eye(n, dtype=B.dtype) + N

    # ---------- 2) Inverse explicite de P (substitution arrière) ----------
    Pinv = np.eye(n, dtype=B.dtype)
    for j in range(1, n):  # colonne courante
        for i in range(j - 1, -1, -1):  # lignes ↓
            S = P[i, i + 1 : j] @ Pinv[i + 1 : j, j] if j - i > 1 else 0.0
            Pinv[i, j] = -(P[i, j] + S)

    return P, Pinv


def compute_M(P_inv, D_sigma, Sigma_W=None):
    """
    Calcule la matrice M = P^{-1} (SigmaSigmaT) (P^{-1})^T, avec
    SigmaSigmaT = D_sigma * Sigma_W * D_sigma.

    Paramètres:
      P: matrice de passage (5x5)
      D_sigma: matrice D_sigma (5x5)
      Sigma_W: matrice de covariance du bruit blanc (par défaut identité)

    Retourne:
      M: matrice M (5x5)
    """
    if Sigma_W is None:
        Sigma_W = np.eye(len(D_sigma))
    SigmaSigmaT = D_sigma @ Sigma_W @ D_sigma.T
    M = P_inv @ SigmaSigmaT @ P_inv.T
    return M


def compute_Kmat(kappa_vec, delta):
    """
    Calcule la matrice K dont les éléments sont
      K_{ij} = (1 - exp(-delta*(kappa_i+kappa_j)))/(kappa_i+kappa_j)

    Paramètres:
      delta: pas de temps (scalaire)
      kappa: dictionnaire avec les valeurs de kappa pour 'c', 'r', 'l', 'm', 'q'

    Retourne:
      K: matrice K (5x5)
    """

    n = len(kappa_vec)
    K = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K[i, j] = compute_K(kappa_vec[i], kappa_vec[j], delta)
    return K


def compute_sigma_epsilon_from_base(factors: dict = None, Sigma_W=None, delta=1):
    """
    Calcule Sigma_epsilon = P (K ⊙ M) P^T à partir des paramètres de base kappa et sigma.

    Paramètres:
      delta: pas de temps (scalaire)
      kappa: dictionnaire avec les valeurs de kappa pour 'c', 'r', 'l', 'm', 'q'
      sigma: dictionnaire avec les valeurs de sigma pour 'c', 'r', 'l', 'm', 'q'
      alpha_m: valeur de alpha^(m)
      Sigma_W: (optionnel) matrice de covariance du bruit blanc (par défaut identité)

    Retourne:
      Sigma_epsilon: la matrice de covariance du bruit
    """

    D_sigma = compute_D_sigma(factors)
    B = compute_B(factors)
    P, P_inv = compute_P(B)
    M = compute_M(P_inv, D_sigma, Sigma_W)

    kappas = [val["value"].kappa for val in factors.values()]

    K = compute_Kmat(kappas, delta)
    # Produit de Hadamard : (K ⊙ M)
    H = K * M
    # Transformation dans la base d'origine
    Sigma_epsilon = P @ H @ P.T
    L_epsilon = np.linalg.cholesky(Sigma_epsilon)
    return Sigma_epsilon, L_epsilon


def simulate_normal_paths_variable_delta(
    delta_seq, N, factors: dict = None, Sigma_W=None
):
    T = len(delta_seq)
    kappas = [val["value"].kappa for val in factors.values()]
    d = len(kappas)
    all_epsilons = np.zeros((N, T, d))

    for t in range(T):
        delta_t = delta_seq[t]

        # 1. Calcul de la covariance exacte d'Ahlgrim
        Sigma_epsilon, L_epsilon = compute_sigma_epsilon_from_base(
            factors=factors, Sigma_W=Sigma_W, delta=delta_t
        )

        # 2. Tirage de bruits N(0, 1) indépendants
        Z = np.random.randn(N, d)

        # 3. Application de la covariance (bruits corrélés mais NON réduits)
        correlated_noise = Z @ L_epsilon.T

        # 4. CORRECTION : Re-standardisation pour forcer la variance à 1.
        # On divise chaque colonne par son écart-type exact.
        # Ainsi, la matrice retrouve des N(0, 1) pures, mais avec la bonne corrélation.
        std_devs = np.sqrt(np.diag(Sigma_epsilon))

        # Sécurité pour éviter la division par zéro si un facteur est constant
        std_devs = np.where(std_devs > 1e-12, std_devs, 1.0)

        all_epsilons[:, t, :] = correlated_noise / std_devs

    # Construction du dict
    noises = {k: all_epsilons[:, :, idx].T for idx, k in enumerate(factors.keys())}

    return noises


def compute_corr(factors: dict = None):
    if not factors:
        return np.array([[]])
    keys = list(factors)
    n = len(keys)
    corr = np.eye(n)

    # Map de couples (type_i, type_j) vers (fonction, args dynamiques)
    _cases = {
        ("OU", "OU"): lambda vi, vj: Correlation_2OU(
            vi, vj, same=(vi == vj)
        ).compute_rho(),
        ("OU", "Va2"): lambda vi, vj: Correlation_OU_2vasicek(
            vi, vj, same_longrate=(vi == vj.taux_long)
        ).compute_rho()[0],
        ("OU", "PC"): lambda vi, vj: Correlation_OU_PC(
            vi, vj, same_inflate=(vi == vj.inflate)
        ).compute_rho()[0],
        ("Va2", "OU"): lambda vi, vj: Correlation_OU_2vasicek(
            vj, vi, same_longrate=(vi.taux_long == vj)
        ).compute_rho()[0],
        ("Va2", "Va2"): lambda vi, vj: Correlation_2_2vasicek(
            vi, vj, same_longrate=(vi.taux_long == vj.taux_long)
        ).compute_rho(),
        ("Va2", "PC"): lambda vi, vj: Correlation_2vasicek_PC(
            vi, vj, same_inflate=(vi.taux_long == vj.inflate)
        ).compute_rho(),
        ("PC", "OU"): lambda vi, vj: Correlation_OU_PC(
            vj, vi, same_inflate=(vi.inflate == vj)
        ).compute_rho()[0],
        ("PC", "Va2"): lambda vi, vj: Correlation_2vasicek_PC(
            vj, vi, same_inflate=(vj.taux_long == vi.inflate)
        ).compute_rho(),
        ("PC", "PC"): lambda vi, vj: Correlation_2_PC(
            vi, vj, same_inflate=(vi.inflate == vj.inflate)
        ).compute_rho(),
    }

    for i in range(n):
        vi = factors[keys[i]]["value"]
        for j in range(i + 1, n):
            vj = factors[keys[j]]["value"]
            case = (vi._type, vj._type)
            if case in _cases:
                corr[i, j] = _cases[case](vi, vj)

    corr = corr + corr.T - np.eye(n)
    return nearest_corr(corr, corr_mat=True)


class Ahlgrim:
    def __init__(self, factors: dict = None, Hds: dict = None, correl=None):
        self.factors = factors
        self.Hds = Hds
        self.correl = correl

    def _prepare_data(self, factors):
        Hds = {
            key: value for key, value in factors.items() if value.get("type") == "Hd"
        }
        factors = {
            key: value
            for key, value in factors.items()
            if value.get("type") in ("OU", "Va2", "PC")
        }
        return factors, Hds

    def compute_correl(self):
        self.correl = compute_corr(self.factors)

    def fit_model(self, input_date):
        # ----------- Préparation des inputs -----------
        factors_mix = input_date.copy()
        path = factors_mix.pop("df_path", None)
        correl = factors_mix.pop("correl", None)
        hds_params = factors_mix.pop("hds_params", None)
        hds_delta = factors_mix.pop("hds_delta", 1)
        nb_latent = factors_mix.pop("nb_latent", 2)
        to_calibrate = factors_mix.pop("to_calibrate", None)

        factors, Hds = self._prepare_data(factors_mix)
        Hds = dict(sorted(Hds.items(), key=lambda item: int(item[0])))

        def sort_factors_ou_last(item):
            key, value = item
            # OU à la fin, sinon tri numérique
            return (value["type"] == "OU", int(key))

        def build_value_model(value, factors):
            params = value.get("params", None)
            if value["type"] == "OU":
                fixed_kappa = None
                if value.get("fixed_kappa") is True:
                    fixed_kappa = 0
                return Ornstein_Uhlenbeck(params=params, fix_kappa=fixed_kappa)
            elif value["type"] == "Va2":
                latent_model = factors[value["latent_key"]]["value"]
                return Two_factor_Vasicek(params=params, taux_long_model=latent_model)
            elif value["type"] == "PC":
                latent_model = factors[value["latent_key"]]["value"]
                return Phillips_curve(params=params, infla_model=latent_model)
            else:
                raise ValueError(f"Type de modèle {value['type']} non reconnu.")

        # ----------- Calibration -----------
        if to_calibrate:
            # Calibrage des facteurs
            keys = list(factors.keys())
            limit = len(keys) ** 2
            it = 0
            while keys and it < limit:
                it += 1
                for cle in keys[:]:
                    value = factors[cle]
                    # Dépendance non satisfaite
                    if (
                        "latent_key" in value
                        and "value" not in factors[value["latent_key"]]
                    ):
                        continue
                    if "ajustement_var" in value and any(
                        "value" not in factors[j] for j in value["ajustement_var"]
                    ):
                        continue

                    df_path = {"index": value["index"], "path": path}
                    ajustement_var = None
                    if "ajustement_var" in value:
                        ajustement_var = [
                            factors[j]["value"] for j in value["ajustement_var"]
                        ][0]
                    value_model = None
                    if value["type"] == "OU":
                        fixed_kappa = None
                        if value.get("fixed_kappa") is True:
                            fixed_kappa = 0
                        value_model = Ornstein_Uhlenbeck(fix_kappa=fixed_kappa)
                        value_model.fit_model(
                            df_path=df_path,
                            ajustement_var=ajustement_var,
                            df_mu_path=value.get("df_mu_path"),
                        )
                    elif value["type"] == "Va2":
                        value_model = Two_factor_Vasicek(
                            taux_long_model=factors[value["latent_key"]]["value"]
                        )
                        value_model.fit_model(
                            df_path=df_path, ajustement_var=ajustement_var
                        )

                    elif value["type"] == "PC":
                        value_model = Phillips_curve(
                            infla_model=factors[value["latent_key"]]["value"]
                        )
                        value_model.fit_model(
                            df_path=df_path, ajustement_var=ajustement_var
                        )
                    factors[cle]["value"] = value_model
                    keys.remove(cle)

            factors = dict(sorted(factors.items(), key=sort_factors_ou_last))
            correl = compute_corr(factors)

            # Calibration Hds
            if Hds:
                p_Hds = len(Hds)
                init_params = (
                    hd_params(nb_latent, p_Hds) if nb_latent > 2 or p_Hds != 2 else None
                )
                value_Hds = HardyMultivariate(init_params)
                df_path_ajustement_var_hds = {
                    key: [
                        {"index": value.get("index"), "path": path},
                        factors.get(value.get("ajustement_var", [None])[0], {}).get(
                            "value"
                        ),
                    ]
                    for key, value in Hds.items()
                }

                value_Hds.fit_model(
                    df_path_ajustement_var=df_path_ajustement_var_hds, max_iter=10
                )

                Hds["value"] = value_Hds

        # ----------- Cas sans calibration -----------
        else:
            factors = dict(sorted(factors.items(), key=lambda item: int(item[0])))
            keys = list(factors.keys())
            limit = len(keys) ** 2
            it = 0
            while keys and it < limit:
                it += 1
                for cle in keys[:]:
                    value = factors[cle]
                    if (
                        "latent_key" in value
                        and "value" not in factors[value["latent_key"]]
                    ):
                        continue
                    value_model = build_value_model(value, factors)
                    factors[cle]["value"] = value_model
                    keys.remove(cle)

            if Hds:

                value_Hds = HardyMultivariate(hds_params)
                value_Hds.delta = hds_delta
                Hds["value"] = value_Hds

        self.Hds = Hds
        self.factors = factors
        self.correl = correl

    # ================================================================
    #  POST-SCENARIO EM CALIBRATION  (Paper Section 4.2)
    # ================================================================

    def _em_extract_structure(self):
        """
        Extract structural information from self.factors needed for the
        multivariate VAR representation.

        Returns
        -------
        info : dict with keys
            factor_keys : list of str
            d            : int (number of factors)
            types        : list of str  ('OU', 'Va2', 'PC')
            va2_to_ou    : dict {i: j}  Va2 index -> its taux_long OU index
            pc_to_ou     : dict {i: j}  PC  index -> its inflate  OU index
            fixed_kappa  : list of bool – True when the factor's κ is
                           fixed at 0 (ABM / GBM), i.e. must NOT be
                           optimised in the M-step.
        """
        fkeys = list(self.factors.keys())
        d = len(fkeys)
        types = []
        va2_to_ou = {}
        pc_to_ou = {}
        fixed_kappa = []

        for i, k in enumerate(fkeys):
            fac = self.factors[k]
            t = fac.get("type", getattr(fac["value"], "_type", "OU"))
            types.append(t)
            if t == "Va2" and "latent_key" in fac:
                j = fkeys.index(fac["latent_key"])
                va2_to_ou[i] = j
            elif t == "PC" and "latent_key" in fac:
                j = fkeys.index(fac["latent_key"])
                pc_to_ou[i] = j
            # Detect fixed κ: either from config dict or from model kappa≈0
            is_fixed = fac.get("fixed_kappa", False) is True
            if not is_fixed and abs(getattr(fac["value"], "kappa", 1.0)) < 1e-14:
                is_fixed = True
            fixed_kappa.append(is_fixed)

        return dict(
            factor_keys=fkeys,
            d=d,
            types=types,
            va2_to_ou=va2_to_ou,
            pc_to_ou=pc_to_ou,
            fixed_kappa=fixed_kappa,
        )

    # ---------- static-like helpers (explicit parameter inputs) ----------

    @staticmethod
    def _build_B_explicit(kappas, alphas_pc, types, va2_to_ou, pc_to_ou):
        """Build drift matrix B from explicit parameter arrays."""
        d = len(kappas)
        B = -np.diag(kappas)
        for i in range(d):
            if types[i] == "Va2" and i in va2_to_ou:
                j = va2_to_ou[i]
                B[i, j] = kappas[i]
            elif types[i] == "PC" and i in pc_to_ou:
                j = pc_to_ou[i]
                B[i, j] = -alphas_pc[i] * kappas[j]
        return B

    @staticmethod
    def _build_Dsigma_explicit(sigmas, alphas_pc, types, pc_to_ou):
        """Build D_sigma matrix from explicit parameter arrays."""
        d = len(sigmas)
        D = np.diag(sigmas.copy())
        for i in range(d):
            if types[i] == "PC" and i in pc_to_ou:
                j = pc_to_ou[i]
                D[i, j] = alphas_pc[i] * sigmas[j]
        return D

    @staticmethod
    def _build_A_vector(
        kappas,
        mus,
        alphas_pc,
        types,
        pc_to_ou,
        scenario_mu_col,
        t_idx,
        T0_idx,
        scenario_post_col,
    ):
        """
        Build the A_{t_n} column vector for a single scenario at a given
        time index.

        Uses **forward-looking** convention (paper footnote):
        callers pass ``t_idx = n`` (the target observation index), so that
        the scenario mean is evaluated at t_n rather than t_{n-1}.  This
        is coherent with ``Ornstein_Uhlenbeck.fit_model`` and
        ``Ornstein_Uhlenbeck.simulate``.

        * Factor with scenario data (index i in ``scenario_post_col``):
          - t <= T0_idx  →  use ``mus[i]`` (historical constant)
          - t >  T0_idx  →  use ``scenario_post_col[i][...]`` (fixed)

        * Factor without scenario data (index i NOT in ``scenario_post_col``):
          - For ALL t  →  use ``mus[i]`` (single constant).

        Parameters
        ----------
        kappas, mus   : np.arrays of length d
        alphas_pc     : dict {i: alpha} for PC factors
        types         : list of factor types
        pc_to_ou      : dict {i: j}
        scenario_mu_col : None  (not used, kept for compat)
        t_idx         : int – time index in [0, T], typically n (forward)
        T0_idx        : int – last historical index
        scenario_post_col : dict {factor_idx: np.array}

        Returns
        -------
        A : np.array of shape (d,)
        """
        d = len(kappas)
        A = np.zeros(d)

        def _mu_at(i, t_idx):
            """Return mu^(z_i)_{t, Delta} for the current scenario.

            Falls back to the constant ``mus[i]`` when the scenario
            lookup has a gap (NaN) for this year.
            """
            if t_idx <= T0_idx:
                return mus[i]
            elif i in scenario_post_col:
                val = scenario_post_col[i][t_idx - T0_idx - 1]
                # NaN guard: missing years in scenario lookup → fallback
                return val if np.isfinite(val) else mus[i]
            else:
                return mus[i]  # no exogenous trend

        for i in range(d):
            if types[i] == "OU":
                # γ(κ)·μ : pour ABM (κ=0), γ=1 → A=μ ; pour OU (κ>0), γ=κ → A=κμ
                A[i] = gamma_coeff(kappas[i]) * _mu_at(i, t_idx)
            elif types[i] == "Va2":
                A[i] = 0.0  # r mean-reverts to l (in B)
            elif types[i] == "PC":
                j = pc_to_ou[i]  # index of inflation OU
                # κ^(m)·μ^(m) + α^(m)·γ(κ^(q))·μ^(q) (papier eq B.7)
                A[i] = kappas[i] * mus[i] + alphas_pc[i] * gamma_coeff(
                    kappas[j]
                ) * _mu_at(j, t_idx)
        return A

    # ---------- factorised helpers (shared by EM, filters, corr) ----------

    @staticmethod
    def _build_var_system(kappas, sigmas, alphas_pc, info, delta, Sigma_W):
        """
        Build the full set of VAR matrices (Eq. B.11–B.13).

        Factorised helper used by ``_em_m_objective``,
        ``_em_reestimate_corr``, ``_filter_multivariate``, and
        ``_filter_univariate_c``.

        Returns
        -------
        var : dict with keys
            P_mat, P_inv, Phi, cn_diag, Sigma_eps, K_mat, D_sig
        """
        types = info["types"]
        va2_to_ou = info["va2_to_ou"]
        pc_to_ou = info["pc_to_ou"]

        B = Ahlgrim._build_B_explicit(kappas, alphas_pc, types, va2_to_ou, pc_to_ou)
        P_mat, P_inv = compute_P(B)
        D_sig = Ahlgrim._build_Dsigma_explicit(sigmas, alphas_pc, types, pc_to_ou)
        M = compute_M(P_inv, D_sig, Sigma_W)
        K_mat = compute_Kmat(kappas, delta)
        Sigma_eps = P_mat @ (K_mat * M) @ P_mat.T

        # Transition: Φ = P·diag(ar(κ,δ))·P⁻¹  (Eq. Phi_def)
        # ar_coeff(0, δ) = 0 (ABM: données i.i.d.)
        ar_diag = np.array([ar_coeff(k, delta) for k in kappas])
        Phi = P_mat @ np.diag(ar_diag) @ P_inv
        # g_{n,j} coefficient: P·diag(Ψ(κ,δ))·P⁻¹  (Eq. cn_def)
        cn_diag = np.diag(np.array([Psi(k, delta) for k in kappas]))

        return dict(
            P_mat=P_mat,
            P_inv=P_inv,
            Phi=Phi,
            cn_diag=cn_diag,
            Sigma_eps=Sigma_eps,
            K_mat=K_mat,
            D_sig=D_sig,
        )

    def _extract_model_params(self, info):
        """
        Read current calibrated parameters from self.factors.

        Factorised helper used by ``calibrate_post_scenario`` (E-step in
        multivariate mode) and ``filter_trajectory``.

        Returns
        -------
        kappas, sigmas : np.arrays (d,)
        mus            : np.array  (d,)
        alphas_pc      : dict {i: alpha}
        """
        fkeys = info["factor_keys"]
        d = info["d"]
        types = info["types"]

        kappas = np.array([self.factors[k]["value"].kappa for k in fkeys])
        sigmas = np.array([self.factors[k]["value"].sigma for k in fkeys])
        mus = np.zeros(d)
        for i, k in enumerate(fkeys):
            m = self.factors[k]["value"]
            if hasattr(m, "mu") and m.mu is not None:
                mus[i] = m.mu
        alphas_pc = {}
        for i, k in enumerate(fkeys):
            if types[i] == "PC":
                alphas_pc[i] = self.factors[k]["value"].alpha
        return kappas, sigmas, mus, alphas_pc

    # ---------- Data loading helpers ----------

    @staticmethod
    def _load_raw_scenario_df(df_mu_path):
        """
        Load raw scenario data from the database and return it together
        with all the metadata needed for further processing.

        Parameters
        ----------
        df_mu_path : dict  {"index": str, "path": str}

        Returns
        -------
        df_sc       : pd.DataFrame  – raw scenario data
        meta        : dict with keys 'date_col', 'mu_col', 'type_data',
                      'filtres', 'ToPass', 'mu_adjusted'
        """
        import pandas as pd
        from gse_engine.db import read_sql_sheet

        dico = read_sql_sheet(df_mu_path["index"], df_mu_path["path"])
        date_col = dico.loc[dico["Refrence"] == "Date", "data"].values[0]
        mu_col = dico.loc[dico["Refrence"] == "Variable", "data"].values[0]
        type_data = dico.loc[dico["Refrence"] == "Type", "data"].values[0]
        DataLeaf = dico.loc[dico["Refrence"] == "DataLeaf", "data"].values[0]
        ToPass = dico.loc[dico["Refrence"] == "ToPass", "data"].values[0]

        mu_adjusted_values = dico.loc[dico["Refrence"] == "mu_adjusted", "data"].values
        mu_adjusted_raw = (
            mu_adjusted_values[0] if len(mu_adjusted_values) > 0 else False
        )
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

        # Filters
        filtres = None
        if ToPass == "false":
            val_to_filtre = dico.loc[dico["Refrence"] == "ToFiltre", "data"].values[0]
            ToFiltre = val_to_filtre.split("-") if pd.notna(val_to_filtre) else None
            val_value_filtre = dico.loc[
                dico["Refrence"] == "ValueFiltre", "data"
            ].values[0]
            ValueFiltre = (
                val_value_filtre.split("-") if pd.notna(val_value_filtre) else None
            )
            filtres = (
                [[tf, vf] for tf, vf in zip(ToFiltre, ValueFiltre)]
                if (ToFiltre is not None and ValueFiltre is not None)
                else [[None, None]]
            )

        df_sc = read_sql_sheet(DataLeaf, df_mu_path["path"], dtype={date_col: str})

        return df_sc, dict(
            date_col=date_col,
            mu_col=mu_col,
            type_data=type_data,
            filtres=filtres,
            ToPass=ToPass,
            mu_adjusted=mu_adjusted,
        )

    @staticmethod
    def _scenario_df_to_lookup(df_sc, meta, ajust_df_sc=None, ajust_meta=None):
        """
        Process a raw scenario DataFrame into a {year: mu} lookup in
        log-variation space, optionally adjusting by a second scenario.

        Replicates the logic of ``Ornstein_Uhlenbeck._prepare_mu_scenario``
        for a single scenario.

        Returns
        -------
        sc_lookup : dict {int_year: float_mu}
        """
        import pandas as pd

        date_col = meta["date_col"]
        mu_col = meta["mu_col"]
        type_data = meta["type_data"]
        filtres = meta["filtres"]
        ToPass = meta["ToPass"]
        mu_adjusted = meta["mu_adjusted"]

        df = df_sc.copy()

        # ── Adjustment by the adjustment variable's scenario ──
        if ajust_df_sc is not None and not mu_adjusted:
            ajust_date = ajust_meta["date_col"]
            ajust_mu = ajust_meta["mu_col"]
            df = adjust_dataframes(
                df_x=df,
                df_z=ajust_df_sc.copy(),
                df_x_date_col=date_col,
                df_z_date_col=ajust_date,
                df_x_var_col=mu_col,
                df_z_var_col=ajust_mu,
            )

        # ── Filtering and interpolation ──
        if ToPass == "false" and filtres is not None:
            df = filtrer_et_interpoler(
                df,
                colonne_date=date_col,
                colonne_valeur=mu_col,
                filtres=filtres,
                start_date=int(df[date_col].iloc[0]),
                mu0=df[mu_col].iloc[0],
            )
            df[date_col] = df[date_col].astype(int)

        # ── Log-variation ──
        df_log = log_variation(df, mu_col, date_col, name=mu_col, type_data=type_data)

        return dict(
            zip(
                df_log[date_col].astype(int),
                df_log[mu_col].astype(float),
            )
        )

    def _prepare_post_scenario_inputs(self, scenario_mu_post, c_key=None, verbose=True):
        """
        Build all inputs needed by the EM loop and by filter_trajectory.

        Factorised from calibrate_post_scenario and filter_trajectory:
        loads observations from self.factors, loads scenario means from
        database paths, computes T0_idx, and aligns everything.

        Parameters
        ----------
        scenario_mu_post : dict of dict
            ``{scenario_name: {factor_key: {"index": ..., "path": ...}}}``
        c_key : str or None
            Factor key of the decarbonation signal.  Required when
            the caller needs ``scenario_mu_c_post`` (univariate mode).
            When ``None``, no c-specific validation or extraction is
            done and ``scenario_mu_c_post`` is returned as ``None``.

        Returns
        -------
        O               : np.ndarray (T+1, d)  – aligned observation matrix
        scenario_post   : list of dicts (length k)
                          scenario_post[j] = {factor_index: np.array}
        scenario_mu_c_post : list of np.arrays, or None if c_key is None
        T0_idx          : int  – last pre-scenario index in O
        has_scenario    : dict {factor_key: bool}
        """
        import pandas as pd

        info = self._em_extract_structure()
        fkeys = info["factor_keys"]
        d = info["d"]
        scenario_names = list(scenario_mu_post.keys())
        k = len(scenario_names)

        # ================================================================
        #  1. Build observation matrix from model dataframes
        # ================================================================
        factor_dfs = {}
        for fk in fkeys:
            m = self.factors[fk]["value"]
            df_f = m.df[[m.year_col, "log_variation"]].copy()
            df_f = df_f.rename(columns={"log_variation": f"obs_{fk}"})
            df_f[m.year_col] = df_f[m.year_col].astype(int)
            factor_dfs[fk] = (df_f, m.year_col)

        ref_key = fkeys[0]
        merged = factor_dfs[ref_key][0].copy()
        ref_date_col = factor_dfs[ref_key][1]

        for fk in fkeys[1:]:
            df_f, yr_col = factor_dfs[fk]
            if yr_col != ref_date_col:
                df_f = df_f.rename(columns={yr_col: ref_date_col})
            merged = pd.merge(merged, df_f, on=ref_date_col, how="inner")

        merged = merged.sort_values(ref_date_col).reset_index(drop=True)
        obs_dates = merged[ref_date_col].values.astype(int)
        O = merged[[f"obs_{fk}" for fk in fkeys]].values  # (T+1, d)

        # ================================================================
        #  2. Build adjustment variable map
        # ================================================================
        ajust_map = {}
        for fk in fkeys:
            fac = self.factors[fk]
            av = fac.get("ajustement_var")
            if av is not None:
                ajust_key = av[0] if isinstance(av, list) else av
                ajust_map[fk] = ajust_key
            else:
                ajust_map[fk] = None

        # ================================================================
        #  3. Load scenario data for each scenario × factor
        # ================================================================
        sc_lookups = {}
        for sc_name in scenario_names:
            sc_paths = scenario_mu_post[sc_name]
            sc_lookups[sc_name] = {}

            raw_cache = {}
            for fk, path_dict in sc_paths.items():
                raw_cache[fk] = self._load_raw_scenario_df(path_dict)

            for fk, path_dict in sc_paths.items():
                df_sc, meta = raw_cache[fk]
                ajust_key = ajust_map.get(fk)
                ajust_df_sc, ajust_meta = None, None
                if ajust_key is not None and ajust_key in raw_cache:
                    ajust_df_sc, ajust_meta = raw_cache[ajust_key]
                sc_lookups[sc_name][fk] = self._scenario_df_to_lookup(
                    df_sc,
                    meta,
                    ajust_df_sc=ajust_df_sc,
                    ajust_meta=ajust_meta,
                )

        # ================================================================
        #  4. Compute T0_idx
        # ================================================================
        all_sc_years = set()
        for sc_name in scenario_names:
            for fk, lookup in sc_lookups[sc_name].items():
                all_sc_years.update(lookup.keys())

        if not all_sc_years:
            raise ValueError("No scenario years could be loaded.")

        first_sc_year = min(all_sc_years)
        T0_idx = None
        for i, yr in enumerate(obs_dates):
            if yr >= first_sc_year:
                T0_idx = i - 1
                break
        if T0_idx is None:
            T0_idx = len(obs_dates) - 1
        T0_idx = max(T0_idx, 0)

        if verbose:
            print(
                f"  T0_idx={T0_idx} "
                f"(obs years {obs_dates[0]}–{obs_dates[-1]}, "
                f"scenario starts {first_sc_year})"
            )

        # ================================================================
        #  5. Align scenario means with observation indices (post-T0)
        # ================================================================
        all_scenario_keys = set()
        for sc_name in scenario_names:
            all_scenario_keys.update(sc_lookups[sc_name].keys())
        has_scenario = {fk: (fk in all_scenario_keys) for fk in fkeys}

        # c_key validation (only when provided, i.e. univariate mode)
        if c_key is not None:
            for sc_name in scenario_names:
                if c_key not in sc_lookups[sc_name]:
                    raise ValueError(
                        f"c_key='{c_key}' must have scenario means in every "
                        f"scenario dict, but scenario '{sc_name}' is missing it."
                    )

        n_post = len(obs_dates) - T0_idx - 1
        post_dates = obs_dates[T0_idx + 1 :]

        scenario_post = []
        scenario_mu_c_post = [] if c_key is not None else None
        c_idx = fkeys.index(c_key) if c_key is not None else None

        for sc_name in scenario_names:
            sp = {}
            for fk, lookup in sc_lookups[sc_name].items():
                fi = fkeys.index(fk)
                mu_arr = np.array(
                    [lookup.get(int(yr), np.nan) for yr in post_dates],
                    dtype=float,
                )
                sp[fi] = mu_arr
            scenario_post.append(sp)
            if c_idx is not None:
                scenario_mu_c_post.append(sp[c_idx])

        if verbose:
            sc_list = [fk for fk, v in has_scenario.items() if v]
            ct_list = [fk for fk, v in has_scenario.items() if not v]
            print(f"  Factors with scenario-varying mu (post-T0 fixed): {sc_list}")
            print(f"  Factors with constant mu (calibrated over [0,T]): {ct_list}")
            print(f"  Number of scenarios: {k}")
            print(f"  Post-T0 observations: {n_post}")

        return O, scenario_post, scenario_mu_c_post, T0_idx, has_scenario, obs_dates

    # ---------- Hamilton filter (generalised, shared) ----------

    def _filter_univariate_c(
        self,
        c_obs,
        kappa_c,
        sigma_c,
        mu_c,
        scenario_mu_c,
        pi0,
        delta,
        T0_idx=-1,
    ):
        """
        Univariate Hamilton filter on c_t (Eq. densite_filtre_c).

        f(c_n | Δ=j, c_{n-1}) = N(
            c_{n-1}·ar(κ^c,δ) + Ψ(κ^c,δ)·γ^c·μ^c_{n-1,j},
            (σ^c)²·Ψ(2κ^c, δ)
        )

        Shared by the EM E-step (T0_idx ≥ 0) and filter_trajectory
        (T0_idx = −1, all steps use scenario means).

        Parameters
        ----------
        c_obs           : np.array (T+1,)
        kappa_c, sigma_c, mu_c : float
        scenario_mu_c   : list of np.arrays (k arrays)
            When T0_idx ≥ 0: each array has length (T − T0_idx),
            covering times T0+1 … T.
            When T0_idx = −1: each array has length T.
        pi0             : np.array (k,)
        delta           : float
        T0_idx          : int  (default −1 = all post-scenario)

        Returns
        -------
        pi_path : np.ndarray (T+1, k)
        log_lik : float
        """
        from scipy.stats import norm as _norm

        k = len(pi0)
        T_plus_1 = len(c_obs)

        exp_kc = ar_coeff(kappa_c, delta)
        var_c = sigma_c**2 * Psi(2.0 * kappa_c, delta)
        std_c = np.sqrt(max(var_c, 1e-30))
        gam_c = gamma_coeff(kappa_c)
        psi_kd_c = Psi(kappa_c, delta)

        pi_path = np.zeros((T_plus_1, k))
        pi_path[0] = pi0.copy()
        pi_f = pi0.copy()
        log_lik = 0.0

        for n in range(1, T_plus_1):
            f_n = np.zeros(k)
            for j in range(k):
                if T0_idx >= 0:
                    # EM mode: pre-scenario uses mu_c, post-scenario uses
                    # scenario_mu_c[j] indexed from 0.
                    # Forward-looking (footnote): μ at date of c_obs[n]
                    if n <= T0_idx:
                        mu_n = mu_c
                    else:
                        idx = n - T0_idx - 1
                        mu_n = (
                            scenario_mu_c[j][idx]
                            if idx >= 0 and idx < len(scenario_mu_c[j])
                            else mu_c
                        )
                else:
                    # Simulation mode: all steps use scenario_mu_c[j]
                    # Forward-looking: μ at date of c_obs[n]
                    mu_n = (
                        scenario_mu_c[j][n]
                        if (scenario_mu_c[j] is not None and n < len(scenario_mu_c[j]))
                        else mu_c
                    )

                mean_c = exp_kc * c_obs[n - 1] + psi_kd_c * gam_c * mu_n
                f_n[j] = _norm.pdf(c_obs[n], loc=mean_c, scale=std_c)

            numerator = pi_f * f_n
            denom = numerator.sum()
            if denom > 1e-300:
                pi_f = numerator / denom
                log_lik += np.log(denom)

            pi_path[n] = pi_f.copy()

        return pi_path, log_lik

    def _filter_multivariate(
        self,
        O,
        kappas,
        sigmas,
        mus,
        alphas_pc,
        info,
        scenario_post,
        pi0,
        delta,
        T0_idx=-1,
    ):
        """
        Full multivariate Hamilton filter (Eq. 3.9 + Appendix Eq. B.14).

        At each step n the conditional density under scenario j is:

            f(o_n | Δ=j, F_{n-1}) = N(o_n ; Φ·o_{n-1} + g_{n,j}, Σ_ε)

        Shared by the EM E-step (T0_idx ≥ 0) and filter_trajectory
        (T0_idx = −1).  Uses log-sum-exp for numerical stability.

        Implementation uses Cholesky decomposition (not explicit Σ⁻¹)
        to avoid numerical breakdown when factor scales differ by
        several orders of magnitude (e.g. emissions σ≈3 vs rates σ≈0.01).

        Parameters
        ----------
        O              : np.ndarray (T+1, d)  observation matrix
        kappas, sigmas : np.arrays (d,)
        mus            : np.array  (d,)
        alphas_pc      : dict {i: alpha}
        info           : structural info dict
        scenario_post  : list of dicts  [{factor_idx: np.array}, ...]
            When T0_idx ≥ 0: arrays have length (T − T0_idx).
            When T0_idx = −1: arrays have length T.
        pi0            : np.array (k,)
        delta          : float
        T0_idx         : int  (default −1)

        Returns
        -------
        pi_path : np.ndarray (T+1, k)   filtered probabilities at each step
        log_lik : float                  marginal log-likelihood
        """
        from scipy.linalg import cho_factor, cho_solve

        d = info["d"]
        types = info["types"]
        pc_to_ou = info["pc_to_ou"]
        k_sc = len(scenario_post)
        T_plus_1 = O.shape[0]

        Sigma_W = self.correl if self.correl is not None else np.eye(d)
        var = self._build_var_system(kappas, sigmas, alphas_pc, info, delta, Sigma_W)
        Phi = var["Phi"]
        cn_diag = var["cn_diag"]
        P_mat = var["P_mat"]
        P_inv = var["P_inv"]
        Sigma_eps = var["Sigma_eps"]

        # ---- Cholesky decomposition (numerically stable) ----
        try:
            cho = cho_factor(Sigma_eps)
        except np.linalg.LinAlgError:
            raise ValueError("Σ_ε is not positive-definite.")
        # log det from Cholesky: det(Σ) = prod(L_ii)^2
        log_det = 2.0 * np.sum(np.log(np.diag(cho[0])))

        pi_path = np.zeros((T_plus_1, k_sc))
        pi_path[0] = pi0.copy()
        log_pi = np.log(np.maximum(pi0, 1e-300))
        log_lik = 0.0

        log_norm_const = -0.5 * (d * np.log(2.0 * np.pi) + log_det)

        for n in range(1, T_plus_1):
            log_f_n = np.empty(k_sc)
            for j in range(k_sc):
                # Forward-looking (footnote): A_{t_n}, not A_{t_{n-1}}
                A_n = self._build_A_vector(
                    kappas,
                    mus,
                    alphas_pc,
                    types,
                    pc_to_ou,
                    None,
                    n,
                    T0_idx,
                    scenario_post[j],
                )
                g_nj = P_mat @ cn_diag @ P_inv @ A_n
                e_n = O[n] - Phi @ O[n - 1] - g_nj
                # Quadratic form via Cholesky solve (stable):
                #   e^T Σ⁻¹ e = e^T (cho_solve(e))
                v = cho_solve(cho, e_n)
                log_f_n[j] = log_norm_const - 0.5 * (e_n @ v)

            # Guard: if all log-densities are -inf, keep previous π
            if np.all(~np.isfinite(log_f_n)):
                pi_path[n] = pi_path[n - 1]
                continue

            # Log-sum-exp: π_n ∝ π_{n-1} ⊙ f_n
            log_num = log_pi + log_f_n
            max_log = np.max(log_num[np.isfinite(log_num)])
            log_denom = max_log + np.log(
                np.sum(
                    np.exp(np.where(np.isfinite(log_num), log_num - max_log, -700.0))
                )
            )

            log_pi = log_num - log_denom
            # Clamp to avoid NaN from -inf - (-inf)
            log_pi = np.where(np.isfinite(log_pi), log_pi, -700.0)
            log_lik += log_denom
            pi_path[n] = np.exp(log_pi)

        return pi_path, log_lik

    # ---------- parameter pack / unpack ----------

    def _em_pack_params(self, info):
        """Pack model parameters into a flat vector for optimisation."""
        fkeys = info["factor_keys"]
        d = info["d"]
        types = info["types"]

        kappas = np.array([self.factors[k]["value"].kappa for k in fkeys])
        sigmas = np.array([self.factors[k]["value"].sigma for k in fkeys])

        extra = []  # mus and alphas (PC)
        extra_meta = []  # (factor_idx, 'mu' | 'alpha')

        for i, k in enumerate(fkeys):
            m = self.factors[k]["value"]
            if types[i] in ("OU", "PC"):
                extra.append(m.mu)
                extra_meta.append((i, "mu"))
            if types[i] == "PC":
                extra.append(m.alpha)
                extra_meta.append((i, "alpha"))

        vec = np.concatenate([kappas, sigmas, np.array(extra)])
        return vec, extra_meta

    def _em_unpack_params(self, vec, info, extra_meta):
        """Unpack flat vector into structured arrays."""
        d = info["d"]
        kappas = vec[:d].copy()
        sigmas = vec[d : 2 * d].copy()

        mus = np.zeros(d)
        alphas_pc = {}

        # Fill mus from the model objects (defaults)
        for i, k in enumerate(info["factor_keys"]):
            m = self.factors[k]["value"]
            if hasattr(m, "mu") and m.mu is not None:
                mus[i] = m.mu

        offset = 2 * d
        for idx, (fi, kind) in enumerate(extra_meta):
            val = vec[offset + idx]
            if kind == "mu":
                mus[fi] = val
            elif kind == "alpha":
                alphas_pc[fi] = val

        # Fill alpha for any PC that wasn't in extra_meta
        for i, k in enumerate(info["factor_keys"]):
            m = self.factors[k]["value"]
            if info["types"][i] == "PC" and i not in alphas_pc:
                alphas_pc[i] = m.alpha if m.alpha is not None else 0.0

        return kappas, sigmas, mus, alphas_pc

    def _em_update_models(self, kappas, sigmas, mus, alphas_pc, info, Sigma_W=None):
        """Push optimised parameters back into the model objects.

        For ABM factors (``info["fixed_kappa"][i] == True``), κ is left
        untouched at 0; only μ and σ are updated.
        """
        for i, k in enumerate(info["factor_keys"]):
            m = self.factors[k]["value"]
            # Ne pas écraser κ pour les facteurs à κ fixé (ABM/GBM)
            if not info["fixed_kappa"][i]:
                m.kappa = float(kappas[i])
            m.sigma = float(sigmas[i])
            if info["types"][i] in ("OU", "PC"):
                m.mu = float(mus[i])
            if info["types"][i] == "PC":
                m.alpha = float(alphas_pc.get(i, m.alpha))

            # --- NOUVELLE PARTIE : Mise à jour des corrélations (rho) ---
            if Sigma_W is not None:
                if info["types"][i] == "Va2" and i in info["va2_to_ou"]:
                    j = info["va2_to_ou"][i]
                    m.rho = float(Sigma_W[i, j])
                elif info["types"][i] == "PC" and i in info["pc_to_ou"]:
                    j = info["pc_to_ou"][i]
                    m.rho = float(Sigma_W[i, j])

    # ---------- M-step objective (Equation 4.5) ----------

    def _em_m_objective(
        self,
        vec,
        info,
        extra_meta,
        O,
        scenario_post,
        pi_smoothed,
        T0_idx,
        delta,
        Sigma_W,
    ):
        """
        Evaluate the M-step criterion (Eq. 4.5):

            sum_j  pi_j  sum_n [ log det(Sigma_eps)
                + (o_n - Phi o_{n-1} - g_n)^T Sigma_eps^{-1}
                  (o_n - Phi o_{n-1} - g_n) ]

        Uses ``_build_var_system`` for the VAR matrix construction.
        """
        d = info["d"]
        types = info["types"]
        pc_to_ou = info["pc_to_ou"]
        k = len(pi_smoothed)
        T_plus_1 = O.shape[0]

        kappas, sigmas, mus, alphas_pc = self._em_unpack_params(vec, info, extra_meta)

        # κ ≥ 0 (κ=0 pour ABM) ; σ > 0
        if np.any(kappas < 0) or np.any(sigmas <= 0):
            return 1e15

        # ---- Build VAR system (factorised) ----
        try:
            var = self._build_var_system(
                kappas, sigmas, alphas_pc, info, delta, Sigma_W
            )
        except Exception:
            return 1e15

        Sigma_eps = var["Sigma_eps"]
        Phi = var["Phi"]
        cn_diag = var["cn_diag"]
        P_mat = var["P_mat"]
        P_inv = var["P_inv"]

        # Cholesky for numerical stability
        from scipy.linalg import cho_factor, cho_solve

        try:
            cho = cho_factor(Sigma_eps)
            log_det = 2.0 * np.sum(np.log(np.diag(cho[0])))
        except (np.linalg.LinAlgError, ValueError):
            return 1e15

        # ---- Evaluate objective ----
        obj = 0.0
        N_steps = T_plus_1 - 1

        for j in range(k):
            wj = pi_smoothed[j]
            if wj < 1e-30:
                continue
            sp_j = scenario_post[j]

            sum_quad = 0.0
            for n in range(1, T_plus_1):
                # Forward-looking (footnote): A_{t_n}
                A_n = self._build_A_vector(
                    kappas, mus, alphas_pc, types, pc_to_ou, None, n, T0_idx, sp_j
                )
                c_n = P_mat @ cn_diag @ P_inv @ A_n
                e_n = O[n] - Phi @ O[n - 1] - c_n
                v = cho_solve(cho, e_n)
                sum_quad += e_n @ v

            obj += wj * (N_steps * log_det + sum_quad)

        return obj

    def _em_reestimate_corr(
        self,
        O,
        kappas,
        sigmas,
        mus,
        alphas_pc,
        info,
        scenario_post,
        pi_smoothed,
        T0_idx,
        delta,
    ):
        """
        Re-estimate Sigma_W from the weighted residuals of the M-step.

        Uses ``_build_var_system`` for the VAR matrix construction.
        """
        d = info["d"]
        types = info["types"]
        pc_to_ou = info["pc_to_ou"]
        k = len(pi_smoothed)
        T_plus_1 = O.shape[0]
        N_steps = T_plus_1 - 1

        Sigma_W = self.correl if self.correl is not None else np.eye(d)
        var = self._build_var_system(kappas, sigmas, alphas_pc, info, delta, Sigma_W)
        Phi = var["Phi"]
        cn_diag = var["cn_diag"]
        P_mat = var["P_mat"]
        P_inv = var["P_inv"]
        D_sig = var["D_sig"]
        K_mat = var["K_mat"]

        # Weighted residual covariance
        S = np.zeros((d, d))
        for j in range(k):
            wj = pi_smoothed[j]
            if wj < 1e-30:
                continue
            sp_j = scenario_post[j]
            for n in range(1, T_plus_1):
                # Forward-looking (footnote): A_{t_n}
                A_n = self._build_A_vector(
                    kappas, mus, alphas_pc, types, pc_to_ou, None, n, T0_idx, sp_j
                )
                c_n = P_mat @ cn_diag @ P_inv @ A_n
                e_n = O[n] - Phi @ O[n - 1] - c_n
                S += wj * np.outer(e_n, e_n)
        S /= N_steps

        # Back out Sigma_W from S
        M_hat = P_inv @ S @ P_inv.T
        K_safe = np.where(np.abs(K_mat) > 1e-30, K_mat, 1e-30)
        M_hat = M_hat / K_safe

        try:
            D_inv = np.linalg.inv(D_sig)
            raw = D_inv @ P_mat @ M_hat @ P_mat.T @ D_inv.T
            diag_sqrt = np.sqrt(np.abs(np.diag(raw)))
            diag_sqrt = np.where(diag_sqrt > 1e-15, diag_sqrt, 1.0)
            Sigma_W = raw / np.outer(diag_sqrt, diag_sqrt)
            Sigma_W = nearest_corr(Sigma_W, corr_mat=True)
        except np.linalg.LinAlgError:
            Sigma_W = self.correl if self.correl is not None else np.eye(d)

        return Sigma_W

    # ---------- main public method ----------

    def calibrate_post_scenario(
        self,
        scenario_mu_post: Dict[str, Dict[str, dict]],
        c_key: str = None,
        n_iter: int = 5,
        delta: float = 1.0,
        optimize_corr: bool = True,
        mode: str = "univariate",
        calibration_depth: str = "medium",
        verbose: bool = True,
        tol: float = 1e-4,
    ):
        """
        EM-based post-scenario calibration (Paper Section 4.2).

        After the factor-by-factor pre-scenario calibration on [0, T0],
        this method refines **all** parameters jointly using the full
        observation period [0, T] and the latent climate scenario Δ.

        Parameters
        ----------
        scenario_mu_post : dict of dict
            ``{scenario_name: {factor_key: {"index": ..., "path": ...}}}``
        c_key : str or None
            Factor key of the decarbonation rate.
        n_iter : int
            Number of EM iterations.
        delta : float
            Constant time step (in years).
        optimize_corr : bool
            If True, re-estimate Sigma_W at each iteration.
        mode : str
            "univariate" or "multivariate".
        calibration_depth : str
            'fast', 'medium', or 'deep'.
        verbose : bool
            Print progress information.
        tol : float
            Convergence tolerance on log-likelihood change.

        Returns
        -------
        result : dict
            'pi'             – final scenario probabilities (np.array)
            'pi_path'        – full probability trajectory (np.array)
            'log_likelihoods'– filter log-lik at each EM iteration
            'Sigma_W'        – final correlation matrix
            'T0_idx'         – computed boundary index
        """
        from scipy.optimize import (
            minimize as _minimize,
            basinhopping,
            differential_evolution,
        )

        if mode not in ("univariate", "multivariate"):
            raise ValueError(
                f"mode must be 'univariate' or 'multivariate', got '{mode}'"
            )
        if mode == "univariate" and c_key is None:
            raise ValueError("c_key is required when mode='univariate'.")

        # ---- structural info ----
        info = self._em_extract_structure()
        fkeys = info["factor_keys"]
        d = info["d"]

        # ---- prepare all inputs automatically ----
        (
            O,
            scenario_post,
            scenario_mu_c_post,
            T0_idx,
            has_scenario,
            obs_dates,
        ) = self._prepare_post_scenario_inputs(
            scenario_mu_post,
            c_key=c_key if mode == "univariate" else None,
            verbose=verbose,
        )

        k = len(scenario_post)
        c_idx = fkeys.index(c_key) if c_key is not None else None

        # --- INTERNAL FUNCTION TO RUN FILTER (Avoid redundancy) ---
        def run_filter(current_pi):
            """Executes the Hamilton filter based on the selected mode and current parameters."""
            if mode == "univariate":
                model_c = self.factors[c_key]["value"]
                return self._filter_univariate_c(
                    c_obs=O[:, c_idx],
                    kappa_c=model_c.kappa,
                    sigma_c=model_c.sigma,
                    mu_c=model_c.mu,
                    scenario_mu_c=scenario_mu_c_post,
                    pi0=current_pi,
                    delta=delta,
                    T0_idx=T0_idx,
                )
            else:  # multivariate
                kappas, sigmas, mus, alphas_pc = self._extract_model_params(info)
                return self._filter_multivariate(
                    O,
                    kappas,
                    sigmas,
                    mus,
                    alphas_pc,
                    info,
                    scenario_post,
                    current_pi,
                    delta,
                    T0_idx=T0_idx,
                )

        # ---- initial scenario probabilities (uniform) ----
        pi = np.ones(k) / k
        Sigma_W = self.correl.copy() if self.correl is not None else np.eye(d)
        log_liks = []

        # ============================================================
        #  EM LOOP
        # ============================================================
        for it in range(n_iter):
            # E-STEP: Expectation
            pi_path, ll = run_filter(pi)
            pi_smoothed = pi_path[-1]
            log_liks.append(ll)

            if verbose:
                print(
                    f"EM iter {it + 1}/{n_iter} | LogLik: {ll:.4f} | Pi: {np.round(pi_smoothed, 4)}"
                )

            # ---- Early stopping ----
            if len(log_liks) >= 2 and abs(log_liks[-1] - log_liks[-2]) < tol:
                if verbose:
                    print(
                        f"  EM converged (ΔLL={abs(log_liks[-1] - log_liks[-2]):.2e} < tol={tol:.2e})"
                    )
                pi = pi_smoothed.copy()
                break

            # M-STEP: Maximization
            pi = pi_smoothed.copy()
            vec0, extra_meta = self._em_pack_params(info)

            # Bounds setup
            lb, ub = [], []
            for i in range(d):
                lb.append(0.0 if info["fixed_kappa"][i] else 1e-6)
                ub.append(0.0 if info["fixed_kappa"][i] else 50.0)
            for i in range(d):
                lb.append(1e-8)
                ub.append(5.0)
            for _, kind in extra_meta:
                lb.append(-10.0)
                ub.append(10.0)
            bounds = list(zip(lb, ub))

            # Parameter dictionary for the 3 modes
            cfg = {
                "fast": (0.05, 100, 1e-8),
                "medium": (10, 0.20),
                "deep": (20, 200, 1.5, 0.9, 0.001),
            }

            if calibration_depth not in cfg:
                raise ValueError(
                    "calibration_depth must be 'fast', 'medium', or 'deep'."
                )

            # Common arguments shared by all 3 algorithms
            opt_args = (
                info,
                extra_meta,
                O,
                scenario_post,
                pi_smoothed,
                T0_idx,
                delta,
                Sigma_W,
            )

            # --- ALGO 1: Fast local search ---
            if calibration_depth == "fast":
                noise, max_it, f_tol = cfg["fast"]
                lower = [b[0] if b[0] is not None else -np.inf for b in bounds]
                upper = [b[1] if b[1] is not None else np.inf for b in bounds]
                vec0_explore = np.clip(
                    vec0 * np.random.uniform(1 - noise, 1 + noise, len(vec0)),
                    lower,
                    upper,
                )
                res = _minimize(
                    self._em_m_objective,
                    vec0_explore,
                    args=opt_args,
                    method="L-BFGS-B",
                    bounds=bounds,
                    options={"maxiter": max_it, "ftol": f_tol},
                )

            # --- ALGO 2: Moderate multi-basin search ---
            elif calibration_depth == "medium":
                hops, step = cfg["medium"]
                kw = {
                    "method": "L-BFGS-B",
                    "bounds": bounds,
                    "args": opt_args,
                    "options": {"maxiter": 100, "ftol": 1e-8},
                }
                res = basinhopping(
                    self._em_m_objective,
                    vec0,
                    minimizer_kwargs=kw,
                    niter=hops,
                    stepsize=step,
                )

            # --- ALGO 3: Massive global search ---
            else:  # deep
                pop, gen, mut, rec, diff_tol = cfg["deep"]
                res = differential_evolution(
                    self._em_m_objective,
                    bounds=bounds,
                    args=opt_args,
                    strategy="best1bin",
                    maxiter=gen,
                    popsize=pop,
                    mutation=(0.5, mut),
                    recombination=rec,
                    tol=diff_tol,
                    x0=vec0,
                    workers=1,
                )

            # --- Parameter retrieval and logging ---
            if verbose:
                algos = {
                    "fast": "L-BFGS-B (Jitter)",
                    "medium": "Basin-Hopping",
                    "deep": "Differential Evolution",
                }
                print(
                    f"         M-step [{calibration_depth.capitalize()} - {algos[calibration_depth]}] obj={res.fun:.4f}"
                )

            kappas_opt, sigmas_opt, mus_opt, alphas_opt = self._em_unpack_params(
                res.x, info, extra_meta
            )

            # 1. On recalcule la matrice de corrélation D'ABORD
            Sigma_W = None
            if optimize_corr:
                Sigma_W = self._em_reestimate_corr(
                    O,
                    kappas_opt,
                    sigmas_opt,
                    mus_opt,
                    alphas_opt,
                    info,
                    scenario_post,
                    pi_smoothed,
                    T0_idx,
                    delta,
                )
                self.correl = Sigma_W
            else:
                Sigma_W = self.correl

                # 2. On pousse les paramètres (y compris rho via Sigma_W) dans les objets
            self._em_update_models(
                kappas_opt, sigmas_opt, mus_opt, alphas_opt, info, Sigma_W=Sigma_W
            )

        # ============================================================
        #  FINAL SYNC: Recalculate filter with optimized parameters
        # ============================================================
        if verbose:
            print("\n[Final Sync] Recalculating pi_path with optimized parameters...")

        pi_path, _ = run_filter(pi)
        pi = pi_path[-1]

        if not optimize_corr:
            self.correl = compute_corr(self.factors)

        for key, factor in self.factors.items():
            factor["value"].compute_residuals()

        return dict(
            pi=pi,
            pi_path=pi_path,
            dates=obs_dates,
            scenarios=list(scenario_mu_post.keys()),
            log_likelihoods=log_liks,
            Sigma_W=self.correl,
            T0_idx=T0_idx,
        )

    # ================================================================
    #  TRAJECTORY FILTERING  (Paper Section 3.3)
    # ================================================================

    def filter_trajectory(
        self,
        simulations: dict,
        path_idx: int,
        scenario_mu_post: Dict[str, Dict[str, dict]],
        c_key: str = None,
        delta: float = 1.0,
        pi0: np.ndarray = None,
        mode: str = "multivariate",
    ):
        """
        Run the Hamilton filter on a simulated trajectory to infer
        real-time scenario probabilities (Paper Section 3.3).

        Two modes available (same as ``calibrate_post_scenario``):
        * **multivariate** (Eq. densite_cond): full d-dimensional density
        * **univariate** (Eq. densite_filtre_c): c_t approximation

        Parameters
        ----------
        simulations : dict
            Output of ``self.simulate()``.
        path_idx : int
            Simulation path index (1-indexed).
        scenario_mu_post : Dict[str, Dict[str, dict]]
            Same format as ``calibrate_post_scenario``.
        c_key : str or None
            Factor key of the decarbonation signal.
            **Required** for ``mode="univariate"``.
            **Optional** for ``mode="multivariate"``.
        delta : float
            Time step in years.
        pi0 : np.ndarray or None
            Initial scenario probabilities (uniform if None).
        mode : str
            ``"multivariate"`` or ``"univariate"``.

        Returns
        -------
        result : dict
            'pi_path'   – np.ndarray (T+1, k)
            'log_lik'   – float
            'dates'     – np.array of years
            'scenarios' – list of scenario names
        """
        import pandas as pd

        if mode not in ("univariate", "multivariate"):
            raise ValueError(
                f"mode must be 'univariate' or 'multivariate', got '{mode}'"
            )
        if mode == "univariate" and c_key is None:
            raise ValueError("c_key is required when mode='univariate'.")

        info = self._em_extract_structure()
        fkeys = info["factor_keys"]
        d = info["d"]
        types = info["types"]

        # ================================================================
        #  1.  Extract simulation trajectory
        # ================================================================
        gse = simulations["others"]
        col_name = f"z_simul_{path_idx}"

        obs_list = []
        sim_dates = None
        for fk in fkeys:
            df_simu = gse[fk]["simu"]
            if col_name not in df_simu.columns:
                raise ValueError(f"Path '{col_name}' not found in factor '{fk}'.")
            obs_list.append(df_simu[col_name].values.astype(float))
            if sim_dates is None:
                raw_dates = df_simu["Date"]
                if hasattr(raw_dates.iloc[0], "year"):
                    sim_dates = pd.to_datetime(raw_dates).dt.year.values
                else:
                    sim_dates = raw_dates.values.astype(int)

        O = np.column_stack(obs_list)  # (T+1, d)
        T_plus_1 = O.shape[0]
        T = T_plus_1 - 1

        # ================================================================
        #  2.  Extract model parameters (uses factorised helper)
        # ================================================================
        kappas, sigmas, mus, alphas_pc = self._extract_model_params(info)

        # ================================================================
        #  3.  Load scenario means (uses factorised helpers)
        # ================================================================
        scenario_names = list(scenario_mu_post.keys())
        k_sc = len(scenario_names)

        ajust_map = {}
        for fk in fkeys:
            av = self.factors[fk].get("ajustement_var")
            if av is not None:
                ajust_key = av[0] if isinstance(av, list) else av
                ajust_map[fk] = ajust_key
            else:
                ajust_map[fk] = None

        sc_lookups = {}
        for sc_name in scenario_names:
            sc_paths = scenario_mu_post[sc_name]
            raw_cache = {}
            for fk, path_dict in sc_paths.items():
                raw_cache[fk] = self._load_raw_scenario_df(path_dict)
            sc_lookups[sc_name] = {}
            for fk, path_dict in sc_paths.items():
                df_sc, meta = raw_cache[fk]
                ajust_key = ajust_map.get(fk)
                ajust_df_sc, ajust_meta = None, None
                if ajust_key is not None and ajust_key in raw_cache:
                    ajust_df_sc, ajust_meta = raw_cache[ajust_key]
                sc_lookups[sc_name][fk] = self._scenario_df_to_lookup(
                    df_sc, meta, ajust_df_sc, ajust_meta
                )

        # ================================================================
        #  4.  Align scenario means with simulation dates
        # ================================================================
        c_idx = fkeys.index(c_key) if c_key is not None else None
        scenario_post = []
        scenario_mu_c = [] if c_idx is not None else None

        for sc_name in scenario_names:
            sp = {}
            for fk, lookup in sc_lookups[sc_name].items():
                fi = fkeys.index(fk)
                # Forward-looking requires arrays of length T+1 (index n
                # up to T), hence sim_dates (not sim_dates[:T]).
                mu_arr = np.array(
                    [lookup.get(int(yr), mus[fi]) for yr in sim_dates],
                    dtype=float,
                )
                sp[fi] = mu_arr
            scenario_post.append(sp)
            if c_idx is not None:
                scenario_mu_c.append(sp.get(c_idx))

        # ================================================================
        #  5.  Initial probabilities
        # ================================================================
        if pi0 is None:
            pi0 = np.ones(k_sc) / k_sc
        pi0 = np.asarray(pi0, dtype=float)
        if len(pi0) != k_sc:
            raise ValueError(
                f"pi0 has {len(pi0)} entries but " f"{k_sc} scenarios were provided."
            )

        # ================================================================
        #  6.  Run the Hamilton filter (uses factorised methods)
        # ================================================================
        # T0_idx = -1: all simulation steps use scenario means
        if mode == "multivariate":
            pi_path, log_lik = self._filter_multivariate(
                O,
                kappas,
                sigmas,
                mus,
                alphas_pc,
                info,
                scenario_post,
                pi0,
                delta,
                T0_idx=-1,
            )
        else:  # univariate
            pi_path, log_lik = self._filter_univariate_c(
                O[:, c_idx],
                kappas[c_idx],
                sigmas[c_idx],
                mus[c_idx],
                scenario_mu_c,
                pi0,
                delta,
                T0_idx=-1,
            )

        return dict(
            pi_path=pi_path,
            log_lik=log_lik,
            dates=sim_dates,
            scenarios=scenario_names,
        )

    def simulate(
        self,
        z0s: dict[str, float],
        pi=None,
        start_date=None,
        df_mu_path: str = None,
        delta=1,
        T=10,
        N=1,
    ):

        # Simulation de Hardy
        gse_Hds = self.Hds.copy()
        if gse_Hds:
            hds_model = gse_Hds.get("value")
            if hds_model is not None and hasattr(hds_model, "initial_values"):
                try:
                    for k in gse_Hds.keys():
                        if k == "value":
                            continue
                        if str(k) in z0s:
                            hds_model.initial_values[str(k)] = float(z0s[str(k)])
                except Exception:
                    pass

            _, _, primes_dfs = gse_Hds["value"].simulate(
                start_date=start_date, T=T, N=N, pi=pi
            )
        else:
            primes_dfs = None

        for cle, value in gse_Hds.copy().items():
            if cle == "value":
                continue

            df_year = aggregate_by_year(primes_dfs[cle], op="sum")
            try:
                if not df_year.empty:
                    df_year = df_year.sort_values("Date").reset_index(drop=True)
                    sim_cols = [c for c in df_year.columns if c != "Date"]
                    df_year.loc[0, sim_cols] = float(z0s.get(cle, 0.0))
            except Exception:
                pass

            gse_Hds[cle]["simu"] = df_year

        start_date = (
            pd.Timestamp.today()
            if start_date is None
            else pd.to_datetime(start_date, dayfirst=True)
        )
        if T is None:
            raise ValueError("T doit être fourni lorsque df_mu_path n'est pas donné.")

        deltas = np.full(T, delta)
        noises = (
            simulate_normal_paths_variable_delta(
                deltas, N, factors=self.factors, Sigma_W=self.correl
            )
            if self.factors
            else None
        )
        gse = copy.deepcopy(self.factors)

        def simul_OU(cle, value):
            result, _, _ = value["value"].simulate(
                z0=z0s[cle],
                start_date=start_date,
                df_mu_path=value.get("df_mu_path"),
                delta=delta,
                T=T,
                N=N,
                deltas_noise_in=[deltas, noises[cle]],
            )
            return result

        for cle, value in copy.deepcopy(gse).items():
            if "simu" in value:
                continue

            _type = value["value"]._type

            if _type == "OU":
                gse[cle]["simu_brute"] = simul_OU(cle, value)
                gse[cle]["simu"] = aggregate_by_year(gse[cle]["simu_brute"])
            elif _type in ("Va2", "PC"):
                latent_key = value["latent_key"]
                if "simu" not in gse[latent_key]:
                    gse[latent_key]["simu_brut"] = simul_OU(latent_key, gse[latent_key])
                    gse[latent_key]["simu"] = aggregate_by_year(
                        gse[latent_key]["simu_brut"]
                    )

                simu_args = dict(start_date=start_date, delta=delta, T=T, N=N)
                if _type == "Va2":
                    simu_args.update(
                        r0=z0s[cle],
                        l0=z0s[latent_key],
                        l_sim_noise_in=[
                            deltas,
                            noises[cle],
                            gse[latent_key]["simu_brut"].copy(),
                        ],
                    )
                else:  # PC
                    simu_args.update(
                        m0=z0s[cle],
                        q0=z0s[latent_key],
                        q_sim_noise_in=[
                            deltas,
                            noises[cle],
                            gse[latent_key]["simu_brut"].copy(),
                        ],
                    )
                gse[cle]["simu_brute"] = value["value"].simulate(**simu_args)[0]
                gse[cle]["simu"] = aggregate_by_year(gse[cle]["simu_brute"])

        return {"Hds": gse_Hds, "others": gse}
