from .ornstein_ulhenbeck import Ornstein_Uhlenbeck
from .philips_curve import Phillips_curve
from .two_factor_vasicek import Two_factor_Vasicek
from gse_engine.ahlgrim.tools import *
import pandas as pd


class Correlation_2OU:
    def __init__(
        self, OU1: Ornstein_Uhlenbeck, OU2: Ornstein_Uhlenbeck, rho=None, same=False
    ):
        """
        Initialise la classe Correlation avec deux modèles de type Ornstein-Uhlenbeck.
        Chaque modèle doit posséder un attribut 'residuals' contenant une colonne "Date" et des valeurs numériques.
        """
        self.OU1 = OU1
        self.OU2 = OU2
        self.rho = rho
        self.same = same

        if self.same:
            if self.OU1 != self.OU1:
                raise ValueError("Les Ous des deux objets sont différents")
            self.rho = 1

    def compute_rho(self):
        """
        Calcule la corrélation de Pearson entre les données des deux modèles.
        Retourne la valeur de corrélation rho.
        """
        if self.same:
            return 1

        # Fusion des résidus des deux modèles sur la colonne "Date"
        data = pd.merge(
            self.OU1.residuals, self.OU2.residuals, on="Date", how="inner"
        ).drop(columns=["Date"])

        # Calcul de la covariance entre les deux séries de résidus
        cov = data.cov().iloc[0, 1]

        # Calcul de rho en le bornant entre -1 et 1
        denominateur = (
            compute_K(self.OU1.kappa, self.OU2.kappa) * self.OU1.sigma * self.OU2.sigma
        )
        if denominateur != 0:
            self.rho = max(-1, min(1, cov / denominateur))
        else:
            self.rho = None  # Evite une division par zéro

        return self.rho


class Correlation_OU_2vasicek:
    def __init__(
        self,
        OU: Ornstein_Uhlenbeck,
        Va2: Two_factor_Vasicek,
        rho_r=None,
        rho_l=None,
        same_longrate=False,
    ):
        """
        Initialise la classe Correlation avec deux modèles de type Ornstein-Uhlenbeck.
        Chaque modèle doit posséder un attribut 'residuals' contenant une colonne "Date" et des valeurs numériques.
        """
        self.OU = OU
        self.Va2 = Va2
        self.rho_r = rho_r
        self.rho_l = rho_l
        self.same_longrate = same_longrate

        if self.same_longrate:
            if self.OU != self.Va2.taux_long:
                raise ValueError("Le taux long est différents")
            self.rho_l = 1
            self.rho_r = self.Va2.rho

    def compute_rho(self, update=True):
        """
        Calcule la corrélation de Pearson entre les données des deux modèles.
        Retourne la valeur de corrélation rho.
        """

        if self.same_longrate:
            return self.rho_r, self.rho_l

        if self.rho_l is None or update:
            self.rho_l = Correlation_2OU(self.OU, self.Va2.taux_long).compute_rho()

        # Fusion des résidus des deux modèles sur la colonne "Date"
        data = pd.merge(
            self.OU.residuals, self.Va2.residuals, on="Date", how="inner"
        ).drop(columns=["Date"])

        # Calcul de la covariance entre les deux séries de résidus
        cov = data.cov().iloc[0, 1]

        a = self.Va2.kappa / (self.Va2.kappa - self.Va2.taux_long.kappa)

        # Calculer les constantes K une seule fois
        K_ir = compute_K(self.OU.kappa, self.Va2.kappa)
        K_il = compute_K(self.OU.kappa, self.Va2.taux_long.kappa)

        # Calcul de rho en le bornant entre -1 et 1
        denominateur = K_ir * self.OU.sigma * self.Va2.sigma
        spread = (
            a * (K_il - K_ir) * self.rho_l * self.OU.sigma * self.Va2.taux_long.sigma
        )
        if denominateur != 0:
            self.rho_r = max(-1, min(1, (cov - spread) / denominateur))
        else:
            self.rho_r = None  # Evite une division par zéro

        return self.rho_r, self.rho_l


class Correlation_OU_PC:
    def __init__(
        self,
        OU: Ornstein_Uhlenbeck,
        PC: Phillips_curve,
        rho_m=None,
        rho_q=None,
        same_inflate=False,
    ):
        """
        Initialise la classe Correlation avec deux modèles de type Ornstein-Uhlenbeck.
        Chaque modèle doit posséder un attribut 'residuals' contenant une colonne "Date" et des valeurs numériques.
        """
        self.OU = OU
        self.PC = PC
        self.rho_m = rho_m
        self.rho_q = rho_q
        self.same_inflate = same_inflate

        if self.same_inflate:
            if self.OU != self.PC.inflate:
                raise ValueError("L'inflation est différents")
            self.rho_q = 1
            self.rho_m = self.PC.rho

    def compute_rho(self, update=True):
        """
        Calcule la corrélation de Pearson entre les données des deux modèles.
        Retourne la valeur de corrélation rho.
        """
        if self.same_inflate:
            return self.rho_m, self.rho_q

        if self.rho_q is None or update:
            self.rho_q = Correlation_2OU(self.OU, self.PC.inflate).compute_rho()

        # Fusion des résidus des deux modèles sur la colonne "Date"
        data = pd.merge(
            self.OU.residuals, self.PC.residuals, on="Date", how="inner"
        ).drop(columns=["Date"])

        # Calcul de la covariance entre les deux séries de résidus
        cov = data.cov().iloc[0, 1]

        b = (self.PC.alpha * self.PC.inflate.kappa) / (
            self.PC.inflate.kappa - self.PC.kappa
        )

        K_im = compute_K(self.OU.kappa, self.PC.kappa)
        K_iq = compute_K(self.OU.kappa, self.PC.inflate.kappa)

        # Calcul de rho en le bornant entre -1 et 1
        denominateur = K_im * self.OU.sigma * self.PC.sigma
        spread = (
            (self.PC.alpha * K_im + b * (K_iq - K_im))
            * self.rho_q
            * self.OU.sigma
            * self.PC.inflate.sigma
        )
        if denominateur != 0:
            self.rho_m = max(-1, min(1, (cov - spread) / denominateur))
        else:
            self.rho_m = None  # Evite une division par zéro

        return self.rho_m, self.rho_q


class Correlation_2_2vasicek:
    def __init__(
        self,
        Va1: Two_factor_Vasicek,
        Va2: Two_factor_Vasicek,
        rho_mr=None,
        rho_ml=None,
        rho_qr=None,
        rho_ql=None,
        same_longrate=False,
    ):
        """
        Initialise la classe Correlation avec deux modèles.
        Chaque modèle doit posséder un attribut 'residuals' contenant une colonne "Date" et des valeurs numériques.
        """
        self.Va1 = Va1
        self.Va2 = Va2
        self.rho_mr = rho_mr
        self.rho_ml = rho_ml
        self.rho_qr = rho_qr
        self.rho_ql = rho_ql
        self.same_longrate = same_longrate

        if self.same_longrate:
            if self.Va1.taux_long != self.Va2.taux_long:
                raise ValueError("Le facteur latent des deux objets sont différents")
            self.rho_ql = 1

    def compute_rho(self, update=True):
        """
        Calcule la corrélation de Pearson entre les résidus des deux modèles.
        Retourne le tuple (rho_rm, rho_rq, rho_lm, rho_lq).
        """
        if not self.same_longrate:
            self.rho_ql = Correlation_2OU(
                self.Va1.taux_long, self.Va2.taux_long
            ).compute_rho()

        if self.rho_ml is None or update:
            # On récupère les corrélations issues des appels à d'autres classes
            self.rho_ml, self.rho_ql = Correlation_OU_2vasicek(
                self.Va1.taux_long, self.Va2, rho_l=self.rho_ql
            ).compute_rho(update=update)
            self.rho_ql = 1 if self.same_longrate else self.rho_ql

        if self.rho_qr is None or update:
            self.rho_qr, _ = Correlation_OU_2vasicek(
                self.Va2.taux_long, self.Va1, rho_l=self.rho_ql
            ).compute_rho(update=update)

        # Fusion des résidus sur la colonne "Date"
        data = pd.merge(
            self.Va1.residuals, self.Va2.residuals, on="Date", how="inner"
        ).drop(columns=["Date"])

        # Calcul de la covariance entre les deux séries de résidus
        cov = data.cov().iloc[0, 1]

        # Calcul des multiplicateurs a et b
        a = self.Va1.kappa / (self.Va1.kappa - self.Va1.taux_long.kappa)
        b = self.Va2.kappa / (self.Va2.kappa - self.Va2.taux_long.kappa)

        # Calcul des constantes K une seule fois
        K_mr = compute_K(self.Va2.kappa, self.Va1.kappa)
        K_qr = compute_K(self.Va2.taux_long.kappa, self.Va1.kappa)
        K_ml = compute_K(self.Va2.kappa, self.Va1.taux_long.kappa)
        K_ql = compute_K(self.Va2.taux_long.kappa, self.Va1.taux_long.kappa)

        # Denominateur pour le calcul de rho_rm
        denominateur = K_mr * self.Va2.sigma * self.Va1.sigma

        # Calcul du spread selon la formule donnée
        spread = (
            a * (K_ml - K_mr) * self.Va2.sigma * self.Va1.taux_long.sigma * self.rho_ml
            + b
            * (K_qr - K_mr)
            * self.Va2.taux_long.sigma
            * self.Va1.sigma
            * self.rho_qr
            + a
            * b
            * (K_mr - K_ml - K_qr + K_ql)
            * self.Va2.taux_long.sigma
            * self.Va1.taux_long.sigma
            * self.rho_ql
        )

        # Calcul de rho_rm avec mise en borne entre -1 et 1
        if denominateur != 0:
            self.rho_mr = max(-1, min(1, (cov - spread) / denominateur))
        else:
            self.rho_mr = None  # Évite la division par zéro
        return self.rho_mr


class Correlation_2vasicek_PC:
    def __init__(
        self,
        Va2: Two_factor_Vasicek,
        PC: Phillips_curve,
        rho_rm=None,
        rho_lm=None,
        rho_rq=None,
        rho_lq=None,
        same_inflate=False,
    ):
        """
        Initialise la classe Correlation avec deux modèles.
        Chaque modèle doit posséder un attribut 'residuals' contenant une colonne "Date" et des valeurs numériques.
        """
        self.PC = PC
        self.Va2 = Va2
        self.rho_rm = rho_rm
        self.rho_lm = rho_lm
        self.rho_rq = rho_rq
        self.rho_lq = rho_lq
        self.same_inflate = same_inflate

        if self.same_inflate:
            if self.Va2.taux_long != self.PC.inflate:
                raise ValueError("Les taux long des deux objets sont différents")
            self.rho_lq = 1

    def compute_rho(self, update=True):
        """
        Calcule la corrélation de Pearson entre les résidus des deux modèles.
        Retourne le tuple (rho_rm, rho_rq, rho_lm, rho_lq).
        """
        if not self.same_inflate:
            self.rho_lq = Correlation_2OU(
                self.Va2.taux_long, self.PC.inflate
            ).compute_rho()

        if self.rho_lm is None or update:
            # On récupère les corrélations issues des appels à d'autres classes
            self.rho_lm, self.rho_lq = Correlation_OU_PC(
                self.Va2.taux_long, self.PC, rho_q=self.rho_lq
            ).compute_rho(update=update)
            self.rho_lq = 1 if self.same_inflate else self.rho_lq

        if self.rho_rq is None or update:
            self.rho_rq, _ = Correlation_OU_2vasicek(
                self.PC.inflate, self.Va2, rho_l=self.rho_lq
            ).compute_rho(update=update)

        # Fusion des résidus sur la colonne "Date"
        data = pd.merge(
            self.PC.residuals, self.Va2.residuals, on="Date", how="inner"
        ).drop(columns=["Date"])

        # Calcul de la covariance entre les deux séries de résidus
        cov = data.cov().iloc[0, 1]

        # Calcul des multiplicateurs a et b
        a = self.Va2.kappa / (self.Va2.kappa - self.Va2.taux_long.kappa)
        b = (self.PC.alpha * self.PC.inflate.kappa) / (
            self.PC.inflate.kappa - self.PC.kappa
        )

        # Calcul des constantes K une seule fois
        K_rm = compute_K(self.Va2.kappa, self.PC.kappa)
        K_rq = compute_K(self.Va2.kappa, self.PC.inflate.kappa)
        K_lm = compute_K(self.Va2.taux_long.kappa, self.PC.kappa)
        K_lq = compute_K(self.Va2.taux_long.kappa, self.PC.inflate.kappa)

        # Denominateur pour le calcul de rho_rm
        denominateur = K_rm * self.PC.sigma * self.Va2.sigma

        # Calcul du spread selon la formule donnée
        spread = (
            K_rm
            * (
                -self.Va2.sigma
                * self.PC.inflate.sigma
                * (b * self.PC.kappa / self.PC.inflate.kappa)
                * self.rho_rq
                - a * self.Va2.taux_long.sigma * self.PC.sigma * self.rho_lm
                + a
                * (b * self.PC.kappa / self.PC.inflate.kappa)
                * self.Va2.taux_long.sigma
                * self.PC.inflate.sigma
                * self.rho_lq
            )
            + K_rq
            * b
            * self.PC.inflate.sigma
            * (
                self.Va2.sigma * self.rho_rq
                - a * self.Va2.taux_long.sigma * self.rho_lq
            )
            + K_lm
            * a
            * self.Va2.taux_long.sigma
            * (
                self.PC.sigma * self.rho_lm
                - (b * self.PC.kappa / self.PC.inflate.kappa)
                * self.PC.inflate.sigma
                * self.rho_lq
            )
            + K_lq
            * a
            * b
            * self.Va2.taux_long.sigma
            * self.PC.inflate.sigma
            * self.rho_lq
        )

        # Calcul de rho_rm avec mise en borne entre -1 et 1
        if denominateur != 0:
            self.rho_rm = max(-1, min(1, (cov - spread) / denominateur))
        else:
            self.rho_rm = None  # Évite la division par zéro

        return self.rho_rm


class Correlation_2_PC:
    def __init__(
        self,
        PC1: Phillips_curve,
        PC2: Phillips_curve,
        rho_mr=None,
        rho_ml=None,
        rho_qr=None,
        rho_ql=None,
        same_inflate=False,
    ):
        """
        Initialise la classe Correlation avec deux modèles.
        Chaque modèle doit posséder un attribut 'residuals' contenant une colonne "Date" et des valeurs numériques.
        """
        self.PC1 = PC1
        self.PC2 = PC2
        self.rho_mr = rho_mr
        self.rho_ml = rho_ml
        self.rho_qr = rho_qr
        self.rho_ql = rho_ql
        self.same_inflate = same_inflate

        if self.same_inflate:
            if self.PC1.inflate != self.PC2.inflate:
                raise ValueError("Les taux long des deux objets sont différents")
            self.rho_ql = 1

    def compute_rho(self, update=True):
        """
        Calcule la corrélation de Pearson entre les résidus des deux modèles.
        Retourne le tuple (rho_rm, rho_rq, rho_lm, rho_lq).
        """
        if not self.same_inflate:
            self.rho_ql = Correlation_2OU(
                self.PC1.inflate, self.PC2.inflate
            ).compute_rho()

        if self.rho_ml is None or update:
            # On récupère les corrélations issues des appels à d'autres classes
            self.rho_ml, self.rho_ql = Correlation_OU_PC(
                self.PC1.inflate, self.PC2, rho_q=self.rho_ql
            ).compute_rho(update=update)
            self.rho_ql = 1 if self.same_inflate else self.rho_ql

        if self.rho_qr is None or update:
            self.rho_qr, _ = Correlation_OU_PC(
                self.PC2.inflate, self.PC1, rho_q=self.rho_ql
            ).compute_rho(update=update)

        # Fusion des résidus sur la colonne "Date"
        data = pd.merge(
            self.PC1.residuals, self.PC2.residuals, on="Date", how="inner"
        ).drop(columns=["Date"])

        # Calcul de la covariance entre les deux séries de résidus
        cov = data.cov().iloc[0, 1]

        # Calcul des multiplicateurs a et b
        a = (self.PC1.alpha * self.PC1.inflate.kappa) / (
            self.PC1.inflate.kappa - self.PC1.kappa
        )
        b = (self.PC2.alpha * self.PC2.inflate.kappa) / (
            self.PC2.inflate.kappa - self.PC2.kappa
        )

        # Calcul des constantes K une seule fois
        K_mr = compute_K(self.PC2.kappa, self.PC1.kappa)
        K_qr = compute_K(self.PC2.inflate.kappa, self.PC1.kappa)
        K_ml = compute_K(self.PC2.kappa, self.PC1.inflate.kappa)
        K_ql = compute_K(self.PC2.inflate.kappa, self.PC1.inflate.kappa)

        # Denominateur pour le calcul de rho_rm
        denominateur = K_mr * self.PC2.sigma * self.PC1.sigma

        # Calcul du spread selon la formule donnée
        spread = (
            (K_mr * (self.PC1.alpha - a) + a * K_ml)
            * self.PC2.sigma
            * self.PC1.sigma
            * self.rho_ml
            + (K_mr * (self.PC2.alpha - b) + b * K_qr)
            * self.PC2.sigma
            * self.PC1.sigma
            * self.rho_qr
            + (
                K_mr
                * (
                    self.PC2.alpha * self.PC1.alpha
                    - a * self.PC2.alpha
                    - b * self.PC1.alpha
                    + a * b
                )
                + a * K_ml * (self.PC2.alpha - b)
                + b * K_qr * (self.PC1.alpha - a)
                + a * b * K_ql
            )
            * self.PC2.sigma
            * self.PC1.sigma
            * self.rho_ql
        )

        # Calcul de rho_rm avec mise en borne entre -1 et 1
        if denominateur != 0:
            self.rho_mr = max(-1, min(1, (cov - spread) / denominateur))
        else:
            self.rho_mr = None  # Évite la division par zéro

        return self.rho_mr
