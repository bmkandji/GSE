from config import PROJECT_ROOT
from gse_engine.ahlgrim.correlations import *
from gse_engine.ahlgrim.hardy import Hardy, HardyMultivariate
from gse_engine.ahlgrim.ornstein_ulhenbeck import Ornstein_Uhlenbeck
from gse_engine.ahlgrim.philips_curve import Phillips_curve
from gse_engine.ahlgrim.two_factor_vasicek import Two_factor_Vasicek

# from modeles.ahlgrim.simulations import Ahlgrim
from gse_engine.ahlgrim.Mmodel import *
from gse_engine.ahlgrim.output import output_GSE
import numpy as np

# data paths
input_data_path = "data/input_data.db"
mu_acpr = "data/ngfs_scenarios.sqlite"

# emmision carbonne

print("inflation sans mu path")

inflation = Ornstein_Uhlenbeck()

inflation.fit_model({"index": "french_inflation_Index", "path": input_data_path})

inflation.summary()


emission = Ornstein_Uhlenbeck(fix_kappa=0)

emission.fit_model({"index": "owid_ghg_co2e_europe_Index", "path": input_data_path})

emission.summary()

simu = emission.simulate(z0=0.0, T=15, N=5, start_date="2025")


# simulation multivariate model
factors = {
    "to_calibrate": True,
    "df_path": input_data_path,
    "nb_latent": 2,
    "1": {
        "type": "Hd",
        "ECONOMY": "Europe",
        "ajustement_var": ["5"],
        "CLASS": "EQUITY_EURO_EQ_INDEX",
        "MEASURE": "PRICE",
        "index": "Euro_Stoxx_50_Index",
        "output": {
            "data": ["simu", "somprod", "lgzc1"],
            "infos": ["ECONOMY", "MEASURE", "CLASS"],
        },
    },
    "2": {
        "type": "Hd",
        "ECONOMY": "Europe",
        "ajustement_var": ["5"],
        "MEASURE": "PRICE",
        "CLASS": "EQUITY_FRANCE_EQ_INDEX",
        "index": "CAC_40_Index",
        "output": {
            "data": ["simu", "somprod", "lgzc1"],
            "infos": ["ECONOMY", "MEASURE", "CLASS"],
        },
    },
    "3": {
        "type": "OU",
        "fixed_kappa": True,
        "df_mu_path": {
            "index": "REMIND_MAgPIE_3_3_4_8_EU_28_Total_GHG_Below_2_C_Index",
            "path": "data/ngfs_scenarios.sqlite",
        },
        "output": {
            "data": ["simu", "somprod"],
            "infos": ["ECONOMY", "MEASURE", "CLASS"],
        },
        "index": "owid_ghg_co2e_europe_Index",
        "ECONOMY": "Europe",
        "MEASURE": "emission",
        "CLASS": "Europe_emission",
    },
    "5": {
        "type": "OU",
        "df_mu_path": {
            "index": "France_Long_term_interest_rate_Below_2_C_Index",
            "path": "data/ngfs_scenarios.sqlite",
        },
        "ajustement_var": ["7"],
        "output": {
            "data": ["simu", "somprod", "lgzc1", "zcs_real"],
            "infos": ["ECONOMY", "MEASURE", "CLASS"],
        },
        "index": "ecb_interest_france_Index",
        "ECONOMY": "Europe",
        "MEASURE": "RATE",
        "CLASS": "LONG_RATES_FRANCE_LT",
    },
    "7": {
        "type": "OU",
        "df_mu_path": {
            "index": "France_Inflation_rate_Below_2_C_Index",
            "path": "data/ngfs_scenarios.sqlite",
        },
        "output": {"data": ["simu", "lgzc1"], "infos": ["ECONOMY", "MEASURE", "CLASS"]},
        "index": "inflation_france_Index",
        "ECONOMY": "Europe",
        "MEASURE": "RATE",
        "CLASS": "INFLN_FRANCE",
    },
}

# calibration du modele Ahlgrim
ahgrim = Ahlgrim()
ahgrim.fit_model(input_date=factors)


facteurs_a_verifier = ["3", "5", "7"]
memoire_avant = {}

print("================ AVANT CALIBRATION ================")
for k in facteurs_a_verifier:
    modele = ahgrim.factors[k]["value"]
    # On utilise getattr pour éviter une erreur si un paramètre n'existe pas
    memoire_avant[k] = {
        "kappa": getattr(modele, "kappa", 0.0),
        "mu": getattr(modele, "mu", 0.0),
        "sigma": getattr(modele, "sigma", 0.0),
    }
    print(
        f"Facteur {k:2} | kappa: {memoire_avant[k]['kappa']:.10f} | mu: {memoire_avant[k]['mu']:.10f} | sigma: {memoire_avant[k]['sigma']:.10f}"
    )

print("\n... Exécution de l'algorithme EM ...\n")


# ── Post-scenario EM calibration ──
# scenario_mu_post: dict of dict
#   {scenario_name: {factor_key: {"index": ..., "path": ...}}}
# Each inner dict maps factor keys to their DB scenario paths.
# The method will:
#   - automatically extract obs from model.df["log_variation"]
#   - automatically compute T0_idx from date alignment
#   - automatically adjust scenarios when ajustement_var is set

scenario_mu_post = {
    "Below_2_C": {
        "3": {
            "index": "REMIND_MAgPIE_3_3_4_8_EU_28_Total_GHG_Below_2_C_Index",
            "path": mu_acpr,
        },
        "5": {
            "index": "France_Long_term_interest_rate_Below_2_C_Index",
            "path": mu_acpr,
        },
        "7": {"index": "France_Inflation_rate_Below_2_C_Index", "path": mu_acpr},
    },
    "Net_Zero_2050": {
        "3": {
            "index": "REMIND_MAgPIE_3_3_4_8_EU_28_Total_GHG_Net_Zero_2050_Index",
            "path": mu_acpr,
        },
        "5": {
            "index": "France_Long_term_interest_rate_Net_Zero_2050_Index",
            "path": mu_acpr,
        },
        "7": {"index": "France_Inflation_rate_Net_Zero_2050_Index", "path": mu_acpr},
    },
    "NDCs": {
        "3": {
            "index": "REMIND_MAgPIE_3_3_4_8_EU_28_Total_GHG_Nationally_Determined_Contributions_NDCs_Index",
            "path": mu_acpr,
        },
        "5": {"index": "France_Long_term_interest_rate_NDCs_Index", "path": mu_acpr},
        "7": {"index": "France_Inflation_rate_NDCs_Index", "path": mu_acpr},
    },
    "Current_Policies": {
        "3": {
            "index": "REMIND_MAgPIE_3_3_4_8_EU_28_Total_GHG_Current_Policies_Index",
            "path": mu_acpr,
        },
        "5": {
            "index": "France_Long_term_interest_rate_Current_Policies_Index",
            "path": mu_acpr,
        },
        "7": {"index": "France_Inflation_rate_Current_Policies_Index", "path": mu_acpr},
    },
}

result = ahgrim.calibrate_post_scenario(
    scenario_mu_post=scenario_mu_post,
    c_key="3",  # decarbonation factor key
    n_iter=5,
    delta=1.0,
    mode="univariate",
<<<<<<< HEAD
=======
    calibration_depth='medium',  # must be 'fast', 'medium', or 'deep'.
>>>>>>> 11e0a64aecfb24bec652e5c607c1119aac5d277f
)

print(
    result["pi_path"], result["dates"]
)  # scenario probabilities at each iteration (n_iter+1, k)

print("\n================ APRÈS CALIBRATION ================")
for k in facteurs_a_verifier:
    modele = ahgrim.factors[k]["value"]
    k_apres = getattr(modele, "kappa", 0.0)
    m_apres = getattr(modele, "mu", 0.0)
    s_apres = getattr(modele, "sigma", 0.0)

    print(
        f"Facteur {k:2} | kappa: {k_apres:.10f} | mu: {m_apres:.10f} | sigma: {s_apres:.10f}"
    )

    # Calcul des différences pour voir les micro-ajustements
    d_kappa = k_apres - memoire_avant[k]["kappa"]
    d_mu = m_apres - memoire_avant[k]["mu"]
    d_sigma = s_apres - memoire_avant[k]["sigma"]

    print(
        f"           | Δ_kap: {d_kappa:+.10f} | Δ_mu: {d_mu:+.10f} | Δ_sig: {d_sigma:+.10f}\n"
    )

# simulation
# initialization of the Ahlgrim model: in practice, we need to access z0 of each factor
z0s = {"1": 2.5, "2": 1.5, "3": -1.5, "5": 0.0, "6": 0.0, "7": 0.0}
# print(simulations)
simulations = ahgrim.simulate(z0s=z0s, pi=None, T=25, N=5, start_date="31/12/2025")
# result["pi"]      → scenario probabilities (np.array)
# result["T0_idx"]  → automatically computed boundary index

# ── Trajectory filtering (Hamilton filter on a simulation path) ──
# Runs the filter from paper Section 3.3 on simulated trajectories
# to infer real-time scenario probabilities
for p in range(1, 6):  # 5 simulated paths
    filt = ahgrim.filter_trajectory(
        simulations=simulations,
        path_idx=p,
        scenario_mu_post=scenario_mu_post,
        c_key="3",
        delta=1.0,
        pi0=result["pi"],  # use EM-calibrated pi as prior
        mode="multivariate",  # full d-dimensional filter (Eq. 4.2)
        # mode="univariate",        # c_t-only approximation (Eq. densite_filtre_c)
    )
    # filt["pi_path"]   → np.ndarray (T+1, k)  probabilities at each year
    # filt["dates"]     → np.array of years
    # filt["scenarios"] → list of scenario names
    # filt["log_lik"]   → marginal log-likelihood of the path
    print(f"Path {p}: final pi = {np.round(filt['pi_path'], 3)}")
