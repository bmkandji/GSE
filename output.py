import pandas as pd
from gse_engine.ahlgrim.tools import *
import copy


class output_GSE:
    def __init__(
        self,
        factors: dict = None,
        Hds: dict = None,
        maturity=np.array([1, 3, 4, 5, 10, 15, 20, 25]),
    ):
        self.factors = factors
        self.Hds = Hds
        self.maturity = maturity
        self.output = None
        self.to_export = None

    def formate_data(self):

        Hds = self.Hds.copy()
        Hds.pop("value", None)
        factors = self.factors.copy()
        factors.update(Hds)
        keys = list(factors)
        limit = len(keys) ** 2
        it = 0

        while keys and it < limit:
            it += 1
            for cle in keys[:]:  # itère sur une copie pour pouvoir remove
                value = factors[cle]
                # dépendances pas prêtes
                if "ajustement_var" in value and any(
                    "formate" not in factors[j] for j in value["ajustement_var"]
                ):
                    continue

                # IMPORTANT: do not keep a reference to value["simu"], as we mutate
                # the working df when building somprod. We want raw simulations to
                # remain unchanged in the final output.
                formate = {"lgzc1": value["simu"].copy()}
                zcs_real, zcs_real_brut, zcs_nom, zcs_nom_brut = {}, {}, {}, {}

                if (
                    value["type"] == "OU" and value.get("fixed_kappa") is None
                ) or value["type"] == "Va2":
                    for i in self.maturity:
                        if value["type"] == "OU":
                            op = lambda q: np.exp(
                                -term_rate(
                                    s=i,
                                    r=q,
                                    l=value["value"].mu,
                                    kappa_r=value["value"].kappa,
                                    kappa_l=1.0,
                                    sigma_r=value["value"].sigma,
                                    sigma_l=0.0,
                                    mu_hist=value["value"].mu,
                                    rho=0.0,
                                    lambda_l=0.0,
                                )
                                * i
                                / 100
                            )
                            zc_i = apply_op_on_df(value["simu"], op=op)
                        else:
                            op = lambda r, l: np.exp(
                                -term_rate(
                                    s=i,
                                    r=r,
                                    l=l,
                                    kappa_r=value["value"].kappa,
                                    kappa_l=value["value"].taux_long.kappa,
                                    sigma_r=value["value"].sigma,
                                    sigma_l=value["value"].taux_long.sigma,
                                    mu_hist=value["value"].taux_long.mu,
                                    rho=value["value"].rho,
                                    lambda_l=value["value"].prime,
                                )
                                * i
                                / 100
                            )
                            zc_i = combine_df_with_op(
                                value["simu"],
                                factors[value["latent_key"]]["simu"],
                                op=op,
                            )
                        zcs_real[i], zcs_real_brut[i] = trans_pose(zc_i), zc_i

                        # ajustements successifs
                        zc_nom_i = zc_i
                        if "ajustement_var" in value:
                            for ajust_key in value["ajustement_var"]:
                                zc_nom_i = combine_df_with_op(
                                    zc_nom_i,
                                    factors[ajust_key]["formate"]["zcs_nom_brut"][i],
                                    op=lambda x, y: x * y,
                                )
                        zcs_nom[i], zcs_nom_brut[i] = trans_pose(zc_nom_i), zc_nom_i
                        if i == 1:
                            formate["lgzc1"] = apply_op_on_df(
                                zc_nom_i, op=lambda x: -100 * np.log(x)
                            )

                    formate.update(
                        {
                            "zcs_real": zcs_real,
                            "zcs_real_brut": zcs_real_brut,
                            "zcs_nom": zcs_nom,
                            "zcs_nom_brut": zcs_nom_brut,
                        }
                    )
                # ajustements finaux sur lgzc1
                for ajust_key in value.get("ajustement_var", []):
                    formate["lgzc1"] = combine_df_with_op(
                        formate["lgzc1"], factors[ajust_key]["formate"]["lgzc1"]
                    )

                # Build cumulative growth index with base 1 at t0.
                # Work on a copy to avoid overwriting lgzc1 (and therefore value["simu"]).
                somprod = formate["lgzc1"].copy()
                if len(somprod.index) > 0:
                    first_idx = somprod.index[0]
                    somprod.loc[first_idx, somprod.columns != "Date"] = 0

                #to_clip = (value["type"] == "Hd") or value["value"].kappa == 0
                somprod = somme_progressive(somprod) #, to_clip=to_clip
                formate["somprod"] = trans_pose(somprod)
                factors[cle]["formate"] = formate
                keys.remove(cle)
        self.output = factors
        return factors

    def to_xlsx(self):
        type_label = {
            "simu": "ins_value",
            "somprod": "cum_value",
            "lgzc1": "wind_cum_value",
            "zcs_real": "zc_real",
            "zcs_nom": "zc_nom",
        }
        export_frames = []

        # Iterate over the keys in self.output
        for key, output_dict in self.output.items():
            frames = []

            # Check if we have a formate key
            if "formate" in output_dict:
                # Process simu and lgzc1
                if "simu" in output_dict and "lgzc1" in output_dict["formate"]:
                    for item in ["simu", "lgzc1"]:
                        if item == "simu":
                            df = trans_pose(output_dict["simu"])
                        else:
                            df = trans_pose(output_dict["formate"]["lgzc1"])
                        df["TYPE"] = type_label[item]
                        df["TERM"] = 0
                        frames.append(df)

                # Process somprod if available
                if "somprod" in output_dict["formate"]:
                    df = output_dict["formate"]["somprod"]
                    df["TYPE"] = type_label["somprod"]
                    df["TERM"] = 0
                    frames.append(df)

                # Process zcs_real and zcs_nom
                for zcs_type in ["zcs_real", "zcs_nom"]:
                    if zcs_type in output_dict["formate"]:
                        for term, term_df in output_dict["formate"][zcs_type].items():
                            temp = term_df.copy()
                            temp["TYPE"] = type_label[zcs_type]
                            temp["TERM"] = term
                            frames.append(temp)

                # If we have frames, concatenate them
                if frames:
                    res = pd.concat(frames, ignore_index=True)

                    # Move TYPE and TERM columns to the front
                    cols = list(res.columns)
                    for special in ["TYPE", "TERM"]:
                        if special in cols:
                            cols.remove(special)
                    res = res[["TYPE", "TERM"] + cols]

                    # Add any info columns
                    for col in ["ECONOMY", "CLASS", "MEASURE", "SIMULATION"]:
                        if col in output_dict:
                            res.insert(0, col, output_dict[col])

                    export_frames.append(res)

        # Concatenate all frames and sort
        if export_frames:
            self.to_export = pd.concat(export_frames, ignore_index=True)
            if (
                "SIMULATION" in self.to_export.columns
                and "TYPE" in self.to_export.columns
                and "TERM" in self.to_export.columns
            ):
                self.to_export.sort_values(
                    ["SIMULATION", "TYPE", "TERM"], inplace=True, ignore_index=True
                )

            # For debugging, save to a standard location
            self.to_export.to_excel("GSE_output/GSE.xlsx", index=False)
            return self.to_export.copy()
        else:
            # If no frames, return an empty DataFrame with appropriate columns
            return pd.DataFrame(columns=["TYPE", "TERM", "SIMULATION", "Date"])
