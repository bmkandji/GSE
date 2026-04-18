import pandas as pd
import numpy as np
import glob
import os
import re

import numpy as np
import pandas as pd


def log_variation(
        df, value_col, date_col, name: str = "log_variation", type_data: str = "brute"
):
    """
    Trie un DataFrame par date et calcule les log-variations d'une colonne donnée.

    La fonction gère deux formats de date dans la colonne date_col :
      - Format "YYYY" pour une date contenant uniquement l'année (ex. "2020")
      - Format "mm-YYYY" pour une date contenant mois et année (ex. "03-1988")

    :param df: DataFrame Pandas
    :param value_col: str, nom de la colonne contenant les valeurs
    :param date_col: str, nom de la colonne contenant les dates
    :param name: str, nom de la colonne à créer pour stocker la variation logarithmique
    :param type_data: str, "brute" pour la log-variation ou "rate" pour garder la valeur brute
    :return: DataFrame trié avec la colonne 'log_variation', (la première ligne est supprimée pour "brute")
    """

    # Copie du DataFrame pour éviter de modifier l'original
    df = df.copy()

    # Créer une colonne temporaire avec la date convertie en datetime
    df["_date_conv"] = df[date_col].apply(
        lambda x: (
            pd.to_datetime(str(x).strip(), format="%Y", errors="coerce")
            if len(str(x).strip()) == 4
            else pd.to_datetime(str(x).strip(), format="%m-%Y", errors="coerce")
        )
    )

    # Trier le DataFrame selon la date convertie
    df_sorted = df.sort_values(by="_date_conv").copy()

    # Nettoyage de type_data pour éviter les erreurs de casse (ex: "Rate" ou " RATE ")
    type_d = str(type_data).strip().lower()

    # Calculer la log-variation
    if type_d == "brute":
        # SÉCURITÉ : On remplace les 0 et valeurs négatives par NaN avant de faire le log.
        # Cela évite le 'RuntimeWarning: invalid value encountered in log'
        safe_values = df_sorted[value_col].where(df_sorted[value_col] > 0, np.nan)

        df_sorted[name] = 100 * np.log(safe_values).diff()
        result = df_sorted.iloc[1:]  # Supprime la première ligne (NaN)

    elif type_d == "rate":
        df_sorted[name] = df_sorted[value_col]
        result = df_sorted

    else:
        raise ValueError(f"type_data doit être 'brute' ou 'rate'. Valeur reçue : '{type_data}'")

    # Supprimer la colonne temporaire et réinitialiser l'index
    result = result.drop(columns=["_date_conv"]).reset_index(drop=True)

    return result


def adjust_dataframes(
    df_x: pd.DataFrame,
    df_z: pd.DataFrame,
    df_x_date_col: str = "Date",
    df_z_date_col: str = "Date",
    df_x_var_col: str = "mu_scenario",
    df_z_var_col: str = "mu_scenario",
) -> pd.DataFrame:
    """
    Docstring for adjust_dataframes

    :param df_x: Dataframe to be adjusted
    :param df_z: Adjustement Dataframe
    """
    if df_z_date_col != df_x_date_col:
        df_z = df_z.rename(columns={df_z_date_col: df_x_date_col})

    # Ensure both date columns have the same dtype (int) before merging
    df_x[df_x_date_col] = df_x[df_x_date_col].astype(int)
    df_z[df_x_date_col] = df_z[df_x_date_col].astype(int)

    df_x_net = pd.merge(
        df_x,
        df_z.rename(columns={df_z_var_col: "adjusted"}),
        on=df_x_date_col,
        how="inner",
    )

    df_x_net[df_x_var_col] = df_x_net[df_x_var_col] - df_x_net["adjusted"]
    df_final = df_x_net[[df_x_date_col, df_x_var_col]]

    return df_final


def get_last_dates(
    df, date_col="Date", value_col="value", output="year", type_data="brute"
):
    """
    Extrait la dernière date de chaque année et la dernière date de chaque trimestre naturel.

    Ajouts :
      - Pour la partie "année" : conversion de la date en datetime avec le format "%Y"
        afin d'obtenir un datetime du type "YYYY-01-01".
      - Pour les deux parties : conversion de la colonne de valeurs en remplaçant les virgules
        par des points et transformation en numérique.
      - Pour le trimestre, le format de date est "mm-YY" (exemple : "03-20" pour mars 2020).

    Variation :
      - output="year"    : retourne uniquement le DataFrame des dernières dates par année.
      - output="quarter" : retourne uniquement le DataFrame des dernières dates par trimestre.
      - output="both"    : retourne un dictionnaire contenant les deux DataFrames sous les clés
                           'year' et 'quarter'.

    :param df: DataFrame Pandas contenant une colonne de dates et une colonne de valeurs.
    :param date_col: Nom de la colonne contenant les dates (par défaut "Date").
    :param value_col: Nom de la colonne contenant les valeurs (par défaut "value").
    :param output: Choix de la sortie ("year", "quarter" ou "both").
    :return: DataFrame ou dictionnaire en fonction du paramètre output.
    """
    df_last_year = None
    df_last_quarter = None
    # Définition de la fonction lambda
    get_extreme_idx = lambda _df, groups, col, type="brute": getattr(
        _df.groupby(groups)[col], f"idx{'max' if type == 'brute' else 'min'}"
    )()

    if output == "year" or output == "both":
        # --- Extraction pour l'année ---
        df_year = df.copy()
        # Supprimer les lignes où la valeur est manquante
        df_year.dropna(subset=[value_col], inplace=True)
        # Conversion de la colonne de valeurs
        df_year[value_col] = (
            df_year[value_col].astype(str).str.replace(",", ".", regex=True)
        )
        df_year[value_col] = pd.to_numeric(df_year[value_col], errors="coerce")

        # Conversion de la colonne de dates en considérant le format jour/mois/année
        df_year[date_col] = pd.to_datetime(
            df_year[date_col], errors="coerce", format='mixed', dayfirst=True
        )
        df_year.dropna(subset=[date_col], inplace=True)

        # Extraire la dernière date de chaque année
        df_last_year = df_year.loc[
            get_extreme_idx(
                df_year, df_year[date_col].dt.year, date_col, type=type_data
            )
        ]
        # Convertir l'année en datetime au format "%Y" (ex: "2020" -> 2020-01-01)
        df_last_year[date_col] = df_last_year[date_col].dt.year
        df_last_year.reset_index(drop=True, inplace=True)

    if output == "quarter" or output == "both":
        # --- Extraction pour le trimestre ---
        df_quarter = df.copy().dropna(subset=[value_col])
        # Conversion de la colonne de valeurs
        df_quarter[value_col] = (
            df_quarter[value_col].astype(str).str.replace(",", ".", regex=True)
        )
        df_quarter[value_col] = pd.to_numeric(df_quarter[value_col], errors="coerce")

        # Conversion de la colonne de dates en datetime (sans dayfirst pour conserver l'information mois)
        df_quarter[date_col] = pd.to_datetime(
            df_quarter[date_col], errors="coerce", dayfirst=True
        )
        df_quarter.dropna(subset=[date_col], inplace=True)

        # Créer une colonne 'quarter_label' pour assigner un label (3, 6, 9, 12) en fonction de l'intervalle de mois
        # (1-3, 4-6, 7-9, 10-12)
        df_quarter["quarter_label"] = ((df_quarter[date_col].dt.month - 1) // 3 + 1) * 3

        # Créer une colonne 'year'
        df_quarter["year"] = df_quarter[date_col].dt.year

        # Groupby par année et quarter_label et récupérer l'indice de la date maximale pour chaque groupe
        df_last_quarter = df_quarter.loc[
            get_extreme_idx(
                df_quarter, ["year", "quarter_label"], date_col, type=type_data
            )
        ]

        # Créer la colonne 'Date' en combinant le label du trimestre et l'année (exemple : "3-2021")
        df_last_quarter["Date"] = (
            df_last_quarter["quarter_label"].astype(str)
            + "-"
            + df_last_quarter["year"].astype(str)
        )

        # Sélectionner uniquement les colonnes 'Date' et value_col, et réinitialiser l'index
        df_last_quarter = df_last_quarter[["Date", value_col]]
        df_last_quarter.reset_index(drop=True, inplace=True)

    # Retour en fonction du paramètre output
    if output == "year":
        return df_last_year
    elif output == "quarter":
        return df_last_quarter
    elif output == "both":
        return {"year": df_last_year, "quarter": df_last_quarter}
    else:
        raise ValueError("Le paramètre output doit être 'year', 'quarter', ou 'both'.")


def Psi(kappa, delta):
    """
    Fonction auxiliaire Ψ(κ, δ) = ∫₀^δ e^{−κs} ds.

    Ψ(κ, δ) = (1 − e^{−κδ}) / κ   si κ > 0
             = δ                      si κ = 0

    C'est l'extension continue de (1 − e^{−κδ})/κ en κ = 0.
    Utilisée dans toutes les discrétisations exactes OU/ABM.
    """
    if abs(kappa * delta) < 1e-12:
        return delta
    return (1.0 - np.exp(-kappa * delta)) / kappa


def gamma_coeff(kappa):
    """
    Coefficient de drift effectif γ(κ) = 𝟙{κ=0} + κ.

    Pour OU (κ > 0) : γ = κ          → Ψ·γ·μ = (1−e^{−κδ})·μ
    Pour ABM (κ = 0) : γ = 1          → Ψ·γ·μ = δ·μ

    Cf. papier équation (3.4).
    """
    return 1.0 if abs(kappa) < 1e-14 else kappa


def ar_coeff(kappa, delta):
    """
    Coefficient autorégressif pour les données en variations.

    Pour OU (κ > 0) : ar = e^{−κδ}   (les données sont des niveaux → AR(1))
    Pour ABM (κ = 0) : ar = 0         (les données sont des incréments i.i.d.)

    Formellement : e^{−κδ} · 𝟙{κ>0}

    SÉMANTIQUE DES DONNÉES :
    _prepare_data() transforme les données brutes en log-variations.
    - Pour OU (κ>0), ces log-variations sont les NIVEAUX du processus
      mean-reverting → la structure AR(1) s'applique.
    - Pour ABM (κ=0), ces log-variations sont les INCRÉMENTS i.i.d.
      du processus sous-jacent → pas d'autorégression.
    """
    if abs(kappa) < 1e-14:
        return 0.0
    return np.exp(-kappa * delta)


def compute_K(kappa_i, kappa_j, delta=1):
    """
    Calcule la valeur de la fonction K donnée par :

    K(κ^(i), κ^(j)) = (1 - exp(-δₙ (κ^(i) + κ^(j)))) / (κ^(i) + κ^(j))

    Paramètres
    ----------
    kappa_i : float
        La valeur de κ^(i).
    kappa_j : float
        La valeur de κ^(j).
    delta_n : float
        La valeur de δₙ.

    Retourne
    --------
    float
        La valeur de K(κ^(i), κ^(j)).

    Remarque
    --------
    Si κ^(i) + κ^(j) est égal à zéro, la limite de l'expression vaut δₙ.
    """
    sum_kappa = kappa_i + kappa_j

    # Vérifier si la somme est proche de zéro pour éviter la division par zéro
    if np.isclose(sum_kappa, 0.0):
        return delta

    return (1 - np.exp(-delta * sum_kappa)) / sum_kappa


def r2_score(y_true, y_pred):
    """
    Calcul du coefficient de détermination R²
    y_true : array-like, valeurs observées
    y_pred : array-like, valeurs prédites
    """
    # Conversion en numpy arrays pour faciliter les calculs
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    # Calcul des sommes des carrés
    ss_res = np.sum((y_true - y_pred) ** 2)  # Somme des carrés des résidus
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)  # Somme des carrés totaux

    # Calcul du R²
    r2 = 1 - (ss_res / ss_tot)
    return r2


def log_likelihood_null(y):
    """
    Calcul de la log-vraisemblance normalisée du modèle nul pour une régression linéaire
    y : array-like, valeurs observées
    """
    # Conversion en numpy array
    y = np.array(y)

    # Moyenne des observations (modèle nul)
    y_mean = np.mean(y)

    # Variance sous le modèle nul (normalisation en n car c'est le MLE)
    sigma2_null = np.var(y, ddof=0)  # ddof=0 pour normalisation en n

    # Calcul de la log-vraisemblance normalisée du modèle nul
    log_likelihood = -0.5 * np.log(2 * np.pi * sigma2_null) - (
        1 / (2 * sigma2_null)
    ) * np.mean((y - y_mean) ** 2)

    return log_likelihood


def term_rate(s, r, l, kappa_r, kappa_l, sigma_r, sigma_l, mu_hist, rho, lambda_l=0):
    """
    Calcule la valeur de P_real(s) = exp[ A(s) - B1(s)*r1(0) - B2(s)*r2(0) ]
    pour un modèle à deux facteurs de type Vasicek / Hull-White étendu.

    Paramètres
    ----------
    s : float
        Écart de temps (T - t) jusqu'à l'échéance.
    r : float
        Valeur initiale du premier facteur de taux, r1(0).
    l : float
        Valeur initiale du second facteur de taux, r2(0).
    kappa_r, kappa_l : float
        Vitesses de retour à la moyenne (ou décroissance exponentielle)
        pour chacun des deux facteurs.
    sigma_r, sigma_l : float
        Volatilités respectives des deux facteurs.
    mu1, mu2 : float
        Moyennes (ou drifts) respectives.
    rho : float
        Corrélation entre les deux facteurs.

    Retour
    ------
    float
        La valeur de P_real(s).
    """

    # Clamp kappa values to avoid division by zero.
    # kappa = 0 means no mean-reversion; the Va2/OU formulas are not defined at
    # exactly 0, so we use a small positive floor.  kappa_r == kappa_l would also
    # create a 0-denominator in the B_l formula, so we offset slightly.
    _EPS = 1e-8
    kappa_r = max(float(kappa_r), _EPS)
    kappa_l = max(float(kappa_l), _EPS)
    if abs(kappa_r - kappa_l) < _EPS:
        kappa_l = kappa_r + _EPS

    r, l, sigma_r, sigma_l, mu_hist = (
        r / 100,
        l / 100,
        sigma_r / 100,
        sigma_l / 100,
        mu_hist / 100,
    )
    nu = mu_hist - lambda_l * (sigma_l / kappa_l)
    # Fonctions B1(s) et B2(s) corrigées selon l'image
    B_r = (1.0 - np.exp(-kappa_r * s)) / kappa_r
    B_l = (kappa_r / (kappa_r - kappa_l)) * (
        (1.0 - np.exp(-kappa_l * s)) / kappa_l - (1.0 - np.exp(-kappa_r * s)) / kappa_r
    )

    # Calcul de A(s) corrigé selon la formule de l’image

    term1 = (B_r - s) * (nu - (sigma_r**2) / (2 * kappa_r**2)) + B_l * nu
    term2 = -(sigma_r**2 * B_r**2) / (4 * kappa_r)
    term3 = (sigma_l**2 / 2) * (
        (s / kappa_l**2)
        - 2 * (B_r + B_l) / (kappa_l**2)
        + (1 / (kappa_r - kappa_l) ** 2)
        * ((1 - np.exp(-2 * kappa_r * s)) / (2 * kappa_r))
        - (2 * kappa_r / (kappa_l * (kappa_r - kappa_l) ** 2))
        * ((1 - np.exp(-(kappa_r + kappa_l) * s)) / (kappa_r + kappa_l))
        + ((kappa_r**2) / (kappa_l**2 * (kappa_r - kappa_l) ** 2))
        * ((1 - np.exp(-2 * kappa_l * s)) / (2 * kappa_l))
    )
    term4 = (rho * sigma_r * sigma_l) * (
        (1 / (kappa_l * kappa_r)) * (s - 2 * B_r - B_l)
        + (1 / (kappa_r - kappa_l))
        * (
            (1 / kappa_l)
            * ((1 - np.exp(-(kappa_l + kappa_r) * s)) / (kappa_l + kappa_r))
            - (1 / kappa_r) * ((1 - np.exp(-2 * kappa_r * s)) / (2 * kappa_r))
        )
    )
    A_s = term1 + term2 + term3 + term4

    # Expression finale :
    return -100 * (A_s - B_r * r - B_l * l) / s


def importer_et_fusionner_csv(folder_path, sep=","):
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    liste_df = []

    for file in csv_files:
        # Lire le fichier en identifiant les colonnes nécessaires
        df = pd.read_csv(
            file, sep=sep, usecols=["Date", "Dernier"], decimal=",", thousands="."
        )
        # Vérifier si la colonne "Date" contient uniquement des années
        if df["Date"].astype(str).str.match(r"^\d{4}$").all():
            df["Date"] = pd.to_datetime(df["Date"], format="%Y", errors="coerce")
        else:
            df["Date"] = pd.to_datetime(df["Date"], format="%d/%m/%Y", errors="coerce")
        df = get_last_dates(
            df, date_col="Date", value_col="Dernier", output="year", type_data="rate"
        )

        # Nettoyage des valeurs et conversion
        df.dropna(subset=["Date", "Dernier"], inplace=True)
        df["Dernier"] = df["Dernier"].astype(float)

        # Extraire le nom du fichier pour la colonne
        nom_fichier = os.path.splitext(os.path.basename(file))[0].replace(" ", "")
        df.rename(columns={"Dernier": f'{nom_fichier.split("-")[-1]}'}, inplace=True)

        liste_df.append(df)

    if not liste_df:
        print("Aucun fichier valide trouvé.")
        return None

    # Fusionner sur la colonne 'Date'
    df_final = liste_df[0]
    for df in liste_df[1:]:
        df_final = pd.merge(df_final, df, on="Date", how="inner")

    df_final.sort_values("Date", inplace=True)
    colonne_a_retrancher = "0"  # La colonne dont on soustrait les valeurs
    colonnes_a_exclure = ["Date", "0"]
    df_final.loc[:, ~df_final.columns.isin(colonnes_a_exclure)] -= df_final[
        colonne_a_retrancher
    ].values[:, None]

    return df_final.drop("0", axis=1)


def transformer_en_liste_df(df):
    # Extraire les noms de colonnes sauf 'Date'
    maturites = df.columns[1:].astype(float)  # Convertir en numérique

    liste_dfs = []

    # Boucle sur chaque ligne du DataFrame
    for i, row in df.iterrows():
        df_temp = pd.DataFrame(
            {
                "Maturite": maturites,
                "Valeur": row[
                    1:
                ].values,  # Exclure la date, prendre les valeurs correspondantes
            }
        )
        # Trier le DataFrame par la colonne 'Maturite'
        df_temp = df_temp.sort_values(by="Maturite")
        liste_dfs.append(df_temp)

    return liste_dfs


def nearest_corr(A, tol=1e-10, max_iter=10000, corr_mat=False):
    """
    Retourne la matrice de corrélation la plus proche de A
    en utilisant une approche itérative.
    """
    # Initialisation : s'assurer que A est symétrique et a 1 sur la diagonale
    A = (A + A.T) / 2
    if corr_mat:
        np.fill_diagonal(A, 1)

    X = A.copy()
    for i in range(max_iter):
        # Projection sur l'ensemble des matrices PSD via décomposition en valeurs propres
        eigvals, eigvecs = np.linalg.eigh(X)
        # Seuil pour éviter de prendre en compte de légères valeurs négatives dues à l'erreur numérique
        eigvals[eigvals < tol] = tol
        X_psd = eigvecs @ np.diag(eigvals) @ eigvecs.T

        # Projection sur l'ensemble des matrices de corrélation : fixer la diagonale à 1
        X_new = X_psd.copy()
        if corr_mat:
            np.fill_diagonal(X_new, 1)

        # Vérifier la convergence
        if np.linalg.norm(X_new - X, ord="fro") < tol:
            break

        X = X_new.copy()
    return X


def aggregate_by_year(
    df: pd.DataFrame, date_col: str = "Date", op: str = "max"
) -> pd.DataFrame:
    """
    Agrège un DataFrame par année selon l'opération spécifiée.

    Paramètres :
      - df : DataFrame d'entrée.
      - date_col : Nom de la colonne contenant les dates.
      - op : Opération à appliquer, parmi 'min', 'max' ou 'sum'.
             * 'min' : retourne la ligne avec la date minimale pour chaque année.
             * 'max' : retourne la ligne avec la date maximale pour chaque année.
             * 'sum' : retourne la somme de toutes les colonnes numériques par année.

    Retourne :
      - Un DataFrame agrégé avec la colonne de date remplacée par l'année.
    """
    # Travailler sur une copie pour ne pas modifier l'original
    df = df.copy()

    # Conversion de la colonne date en datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Création d'une colonne 'year' extraite de la date
    df["year"] = df[date_col].dt.year

    if op == "min":
        # Récupérer l'index de la ligne avec la date minimale pour chaque année
        idx = df.groupby("year")[date_col].idxmin()
        result = df.loc[idx].copy()
    elif op == "max":
        # Récupérer l'index de la ligne avec la date maximale pour chaque année
        idx = df.groupby("year")[date_col].idxmax()
        result = df.loc[idx].copy()
    elif op == "sum":
        # Pour éviter la somme sur la colonne date (datetime), on la supprime
        result = (
            df.drop(columns=[date_col])
            .groupby("year")
            .sum(numeric_only=True)
            .reset_index()
        )
    else:
        raise ValueError("L'argument 'op' doit être 'min', 'max' ou 'sum'.")

    # Remplacer (ou ajouter) la colonne date par l'année
    result[date_col] = result["year"]
    result.drop(columns=["year"], inplace=True)

    # Optionnel : réorganiser les colonnes pour que date_col apparaisse en première position
    cols = [date_col] + [col for col in result.columns if col != date_col]
    result = result[cols]

    return result.drop(columns=["mu"], errors="ignore", inplace=False)


def combine_df_with_op(
    df_x: pd.DataFrame, df_z: pd.DataFrame, op=lambda x, y: x + y
) -> pd.DataFrame:
    """
    Combine deux DataFrames en alignant sur la colonne 'Date' (année sous forme d'entier).
    Pour chaque colonne de df_x (autre que 'Date'), la fonction extrait le dernier token
    (séparé par '_') et cherche dans df_z la colonne dont le nom se termine par ce même nombre.
    Ensuite, elle applique l'opération 'op' sur ces deux colonnes.

    Paramètres :
      - df_x : DataFrame contenant une colonne 'Date' et d'autres colonnes dont le nom se termine par un nombre.
      - df_z : DataFrame contenant une colonne 'Date' et d'autres colonnes dont le nom se termine par un nombre.
      - op   : Fonction prenant deux Series (ex : lambda x, y: x - y, x + y, x * y, etc.)

    Retourne :
      - Un DataFrame avec la colonne 'Date' et, pour chaque numéro commun, une colonne nommée par ce numéro contenant le résultat de op.
    """
    # Travailler sur des copies pour ne pas modifier les DataFrames d'origine
    df_x = df_x.copy()
    df_z = df_z.copy()

    # S'assurer que la colonne 'Date' est bien un entier (année)
    df_x["Date"] = df_x["Date"].astype(int)
    df_z["Date"] = df_z["Date"].astype(int)

    # Fusionner les DataFrames sur 'Date'
    merged = pd.merge(df_x, df_z, on="Date", how="inner", suffixes=("_x", "_z"))

    # Construire un dictionnaire pour associer le numéro final aux colonnes de df_z dans le merged
    dz_map = {}
    for col in df_z.columns:
        if col == "Date":
            continue
        # Extraction du dernier token séparé par '_'
        tokens = col.split("_")
        if tokens and tokens[-1].isdigit():
            num = tokens[-1]
            # Dans merged, la colonne provenant de df_z peut porter le suffixe '_z'
            if col in merged.columns:
                dz_map[num] = col
            elif f"{col}_z" in merged.columns:
                dz_map[num] = f"{col}_z"
            else:
                dz_map[num] = col
        else:
            raise ValueError(
                f"La colonne '{col}' de df_z ne se termine pas par un nombre."
            )

    # Préparer le DataFrame résultat en conservant la colonne 'Date'
    df_result = pd.DataFrame()
    df_result["Date"] = merged["Date"]

    # On accumule les nouvelles colonnes dans un dictionnaire
    new_cols = {}

    for col in df_x.columns:
        if col == "Date":
            continue
        tokens = col.split("_")
        if tokens and tokens[-1].isdigit():
            num = tokens[-1]
            # Dans merged, la colonne provenant de df_x peut porter le suffixe '_x'
            if col in merged.columns:
                col_merged_x = col
            elif f"{col}_x" in merged.columns:
                col_merged_x = f"{col}_x"
            else:
                col_merged_x = col
            if num not in dz_map:
                raise ValueError(
                    f"Aucune colonne correspondante dans df_z pour le numéro '{num}'."
                )
            col_merged_z = dz_map[num]
            # Appliquer l'opération et stocker la série résultante dans le dictionnaire
            new_cols[num] = op(merged[col_merged_x], merged[col_merged_z])
        else:
            raise ValueError(
                f"La colonne '{col}' de df_x ne se termine pas par un nombre."
            )

    # Créer un DataFrame à partir des nouvelles colonnes accumulées
    df_new = pd.DataFrame(new_cols, index=merged.index)

    # Concaténer avec la colonne "Date" pour obtenir le DataFrame final
    df_result = pd.concat([merged[["Date"]], df_new], axis=1)

    return df_result


def apply_op_on_df(df: pd.DataFrame, op=lambda x: x**2) -> pd.DataFrame:
    """
    Applique une opération unitaire (par défaut le carré) à toutes les colonnes numériques d'un DataFrame,
    en laissant inchangée la colonne 'Date'. Pour chaque colonne transformée, le nouveau nom sera
    le nombre extrait du nom original à l'aide d'une expression régulière.

    Paramètres :
      - df : DataFrame qui doit contenir une colonne 'Date'
      - op : fonction unitaire à appliquer sur les autres colonnes (par défaut, le carré)

    Retourne :
      - Un DataFrame avec la colonne 'Date' inchangée et, pour chaque autre colonne,
        une nouvelle colonne nommée d'après le nombre extrait contenant le résultat de l'opération.
    """
    # Vérifier la présence de la colonne 'Date'
    if "Date" not in df.columns:
        raise ValueError("Le DataFrame doit contenir une colonne 'Date'.")

    # Créer un DataFrame résultat en conservant 'Date'
    df_result = pd.DataFrame()
    df_result["Date"] = df["Date"].astype(int)

    new_cols = {}

    # Pour chaque colonne hors 'Date', extraire le nombre avec une regex et appliquer l'opération
    for col in df.columns:
        if col == "Date":
            continue

        # Chercher le premier nombre dans le nom de la colonne
        match_num = re.findall(r"(\d+)", col)
        if match_num:
            num = match_num[-1]  # dernier nombre trouvé
            if op is None:
                new_cols[num] = df[col]
            else:
                new_cols[num] = df[col].apply(op)
        else:
            raise ValueError(f"Aucun nombre trouvé dans le nom de la colonne '{col}'.")

    # Créer un DataFrame pour les nouvelles colonnes
    df_new = pd.DataFrame(new_cols, index=df.index)

    # Concaténer avec la colonne 'Date'
    df_result = pd.concat([df[["Date"]], df_new], axis=1)

    return df_result


def somme_progressive(df: pd.DataFrame, to_clip: bool = False) -> pd.DataFrame:
    """
    Calcule la somme cumulative progressive d'un DataFrame (hors colonne "Date"),
    divise par 100 et applique l'exponentielle.

    Option :
      - to_clip (bool) : Si True, limite les rendements extrêmes pour éviter l'explosion.
    """
    date_col = None
    # 1. On sépare les données de calcul et la Date dès le début pour éviter les bugs
    if "Date" in df.columns:
        date_col = df["Date"].copy()
        df_calc = df.drop(columns=["Date"]).copy()
    else:
        df_calc = df.copy()

    # 2. On applique le clipping UNIQUEMENT si l'option est activée
    if to_clip:
        # Comme l'entrée est en %, 40 correspond à 40% (et non 0.40)
        limite_haute = 40.0
        limite_basse = -40.0
        # On clip uniquement les colonnes numériques
        df_calc = df_calc.clip(lower=limite_basse, upper=limite_haute)

    # 3. Calcul de la somme cumulative
    df_cum = df_calc.cumsum()

    # 4. Appliquer la transformation : division par 100 et exponentielle
    # L'utilisation de np.exp(df_cum / 100) est plus rapide que .apply()
    df_transformed = np.exp(df_cum / 100)

    # 5. Réinsérer la colonne Date si elle existait
    if "Date" in df.columns:
        df_transformed.insert(0, "Date", date_col)

    return df_transformed


def cantor_pairing(a: int, b: int) -> int:
    """
    Applique la fonction d'appariement de Cantor à un couple (a, b)
    et retourne un unique entier naturel.

    La formule utilisée est :
      π(a, b) = ((a + b) * (a + b + 1)) // 2 + b
    """
    return ((a + b) * (a + b + 1)) // 2 + b


def cantor_pairing3(a: int, b: int, c: int) -> int:
    """
    Applique la fonction d'appariement de Cantor de manière récursive
    pour associer un triplet (a, b, c) à un unique entier naturel.

    On définit :
      π_3(a, b, c) = π(a, π(b, c))
    """
    return cantor_pairing(a, cantor_pairing(b, c))


def filtrer_et_interpoler(
    df, filtres, colonne_date, colonne_valeur, start_date=None, mu0=None
):
    """
    filtres : liste de tuples (colonne_filtre, valeur_filtre)
    start_date : année de départ (int ou None)
    z0 : valeur initiale à start_date (float ou None)
    """
    df_travail = df.copy()

    # Application des filtres
    for colonne, valeur in filtres:
        if colonne is None or pd.isna(colonne) or valeur is None or pd.isna(valeur):
            continue
        df_travail = df_travail[df_travail[colonne] == valeur]

    # Sélection et nettoyage
    df_filtre = df_travail[[colonne_date, colonne_valeur]].copy()
    df_filtre[colonne_date] = df_filtre[colonne_date].astype(int)
    df_filtre = df_filtre.drop_duplicates(subset=colonne_date).sort_values(colonne_date)

    # Ajout de la ligne start_date/z0 si les deux sont fournis
    if start_date is not None and mu0 is not None:
        # Supprime TOUTES les lignes où colonne_date == start_date
        df_filtre = df_filtre[df_filtre[colonne_date] != start_date]
        # Ajoute UNE ligne avec start_date et z0
        ligne_start = {colonne_date: start_date, colonne_valeur: mu0}
        df_filtre = pd.concat(
            [pd.DataFrame([ligne_start]), df_filtre], ignore_index=True
        )
        # Trie les dates
        df_filtre = df_filtre.sort_values(colonne_date)
        # Garde toutes les années >= start_date (y compris celle ajoutée)
        df_filtre = df_filtre[df_filtre[colonne_date] >= start_date]

    # Générer toutes les années du min au max des dates restantes
    all_years = range(df_filtre[colonne_date].min(), df_filtre[colonne_date].max() + 1)
    df_full = pd.DataFrame({colonne_date: all_years})

    # Jointure et interpolation
    df_full = df_full.merge(df_filtre, on=colonne_date, how="left")
    df_full[colonne_valeur] = df_full[colonne_valeur].interpolate(method="linear")

    return df_full


def check(value1, value2, bool=True):
    if bool:
        if value1 != value2:
            raise ValueError("Le facteur latent des deux objets sont différents")
    pass


def trans_pose(df):
    """
    – Indexe df sur la colonne 'Date'
    – Retire le nom de l’index
    – Transpose, remet l’index en colonne 'SIMULATION'
    """

    df = apply_op_on_df(df, op=None)
    df = df.set_index("Date")
    df.index.name = None
    df = df.T.reset_index().rename(columns={"index": "SIMULATION"})
    return df


def hd_params(
    n,
    p,
    mu_base=0,
    mu_gap=10,
    mu_jitter=2,
    sigma_base=100,
    sigma_jitter=10,
    p_min=0.80,
    p_max=0.99,
    seed=None,
):
    """
    - Proba de rester alignée avec la vol (plus la vol est forte, plus la proba de rester est grande).
    """
    rng = np.random.default_rng(seed)
    # Moyennes avec bruit
    mu = np.array(
        [
            np.ones(p) * (mu_base + i * mu_gap) + rng.normal(0, mu_jitter, size=p)
            for i in range(n)
        ]
    )
    # Sigma diagonale, variance variable
    Sigma = np.array(
        [np.diag(sigma_base + rng.normal(0, sigma_jitter, size=p)) for _ in range(n)]
    )
    # Calcule la "force" de la vol par état (moyenne des variances diagonales)
    vols = Sigma.diagonal(axis1=1, axis2=2).mean(axis=1)
    # Normalisation sur [p_min, p_max]
    vols_scaled = (vols - vols.min()) / (vols.max() - vols.min() + 1e-12)
    p_stay = p_min + (p_max - p_min) * vols_scaled  # vecteur taille n

    # Construction de la matrice de transition alignée avec les vols
    P = np.zeros((n, n))
    for i in range(n):
        P[i, i] = p_stay[i]
        P[i, :] += (1 - p_stay[i]) / (n - 1)
        P[i, i] = p_stay[i]  # Corrige la diagonale qui a été augmentée
    # (option : renormalise pour être robuste numériquement)
    P = P / P.sum(axis=1, keepdims=True)

    pi = np.ones(n) / n
    return mu, Sigma, P, pi


def annualize_mu_path(mu_df, start_year, years, year_col=None):
    """
    Annualise (interpole linéairement) un DataFrame mu_path sur une grille annuelle.

    Cette fonction est conçue pour gérer les scénarios climatiques (mu_paths) qui peuvent
    avoir des points de données non-annuels (ex: 2020, 2025, 2030, etc.).
    Elle crée une grille annuelle complète et interpole les valeurs manquantes.

    Paramètres:
      mu_df: DataFrame contenant le mu_path avec une colonne d'années et des colonnes de valeurs
      start_year: int, année de départ de la simulation (ex: 2025)
      years: int, nombre d'années à simuler (ex: 20 pour 2025-2045)
      year_col: str ou None, nom de la colonne contenant les années
                Si None, détecte automatiquement "Year", "year", "Date", ou "date"

    Retourne:
      DataFrame annualisé avec exactement (years + 1) lignes, de start_year à start_year+years inclus

    Raises:
      ValueError: si la colonne année n'est pas trouvée, si les données sont incohérentes,
                 ou si l'intervalle demandé dépasse les données disponibles

    Exemple:
      >>> # mu_path original: 2020, 2025, 2030, 2035, 2040, 2045
      >>> # start_year=2025, years=20
      >>> result = annualize_mu_path(df, 2025, 20)
      >>> # result contient 21 lignes: 2025, 2026, ..., 2045
    """
    if mu_df is None or mu_df.empty:
        raise ValueError("mu_df est vide ou None")

    # Copie pour éviter de modifier l'original
    df = mu_df.copy()

    # Détection automatique de la colonne année
    if year_col is None:
        candidates = ["Year", "year", "Date", "date"]
        for col in candidates:
            if col in df.columns:
                year_col = col
                break
        if year_col is None:
            raise ValueError(
                f"Colonne année non trouvée. Colonnes disponibles: {df.columns.tolist()}"
            )

    if year_col not in df.columns:
        raise ValueError(
            f"Colonne '{year_col}' introuvable. Colonnes disponibles: {df.columns.tolist()}"
        )

    # Nettoyage et parsing des années
    # Convertir en string d'abord pour gérer les formats mixtes
    df[year_col] = df[year_col].astype(str).str.strip()

    # Parser les années (gérer format datetime ou string)
    def parse_year(val):
        try:
            # Si c'est un datetime, extraire l'année
            if "-" in val:
                return int(val.split("-")[0])
            return int(val)
        except (ValueError, AttributeError):
            return None

    df["_year_parsed"] = df[year_col].apply(parse_year)

    # Retirer les lignes avec années non parseables
    invalid_years = df["_year_parsed"].isna()
    if invalid_years.any():
        print(f"Warning: {invalid_years.sum()} ligne(s) avec années invalides retirées")
        df = df[~invalid_years].copy()

    if df.empty:
        raise ValueError("Aucune année valide trouvée dans mu_path")

    # Trier par année
    df = df.sort_values("_year_parsed").reset_index(drop=True)

    # Vérifier l'intervalle disponible
    min_year = int(df["_year_parsed"].min())
    max_year = int(df["_year_parsed"].max())
    end_year = start_year + years

    if start_year < min_year:
        raise ValueError(
            f"start_year ({start_year}) est avant la première année disponible ({min_year})"
        )

    if end_year > max_year:
        raise ValueError(
            f"Horizon demandé ({start_year}-{end_year}) dépasse les données disponibles "
            f"(jusqu'à {max_year}). Réduisez 'years' ou utilisez un mu_path plus long."
        )

    # Utiliser _year_parsed comme index pour l'interpolation
    df.set_index("_year_parsed", inplace=True)

    # Créer la grille annuelle complète
    annual_grid = list(range(start_year, end_year + 1))

    # Identifier les colonnes numériques à interpoler (exclure la colonne année originale)
    # On ne garde que les colonnes numériques, pas la colonne year_col originale
    cols_to_keep = [col for col in df.columns if col != year_col]
    numeric_cols = df[cols_to_keep].select_dtypes(include=[np.number]).columns.tolist()

    # Union des années existantes et de la grille annuelle
    all_years = sorted(set(df.index.tolist() + annual_grid))

    # Reindexer sur toutes les années
    df_reindexed = df.reindex(all_years)

    # Interpoler linéairement les colonnes numériques
    for col in numeric_cols:
        # Utiliser 'linear' avec limit_direction='both' pour interpoler uniquement à l'intérieur
        df_reindexed[col] = df_reindexed[col].interpolate(method="linear")

    # Extraire seulement la grille annuelle demandée
    df_annual = df_reindexed.loc[annual_grid, numeric_cols].copy()

    # Vérifier qu'il n'y a pas de NaN
    if df_annual.isna().any().any():
        nan_cols = df_annual.columns[df_annual.isna().any()].tolist()
        raise ValueError(
            f"Des valeurs NaN subsistent après interpolation dans les colonnes: {nan_cols}"
        )

    # Renommer l'index avant de le reset pour avoir le bon nom de colonne
    df_annual.index.name = year_col

    # Reset index pour remettre l'année en colonne avec le bon nom
    df_annual = df_annual.reset_index()

    # Réordonner les colonnes pour mettre year_col en premier
    cols = [year_col] + [col for col in df_annual.columns if col != year_col]
    df_annual = df_annual[cols]

    return df_annual


def fusionner_xlsx(dossier, fichier_sortie="xlsx_aggree.xlsx"):
    fichiers = [f for f in os.listdir(dossier) if f.endswith(".xlsx")]
    writer = pd.ExcelWriter(os.path.join(dossier, fichier_sortie), engine="openpyxl")

    feuille_idx = {}  # Pour éviter les doublons de noms de feuilles

    for fichier in fichiers:
        chemin = os.path.join(dossier, fichier)
        xls = pd.ExcelFile(chemin)
        for feuille in xls.sheet_names:
            nom = feuille
            # Pour éviter les doublons de noms de feuille
            while nom in feuille_idx:
                feuille_idx[nom] += 1
                nom = f"{feuille}_{feuille_idx[feuille]}"
            feuille_idx[feuille] = feuille_idx.get(feuille, 0)
            df = pd.read_excel(chemin, sheet_name=feuille)
            df.to_excel(
                writer, sheet_name=nom[:31], index=False
            )  # Excel limite les noms de feuilles à 31 caractères

    writer.close()
