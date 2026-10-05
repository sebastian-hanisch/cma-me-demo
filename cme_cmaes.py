"""CMA-ES-Emitter: die Suchverteilung eines einzelnen Emitters von CMA-ME und CMA-MAE.

Umsetzung nach dem CMA-ES-Tutorial von Hansen (2016): Mittelwert `m`, Schrittweite `sigma`, volle Kovarianzmatrix `C` (Eigenzerlegung `B`, `D` träge nachgeführt), Evolutionspfade `pc` und `ps`, logarithmische Gewichte der besten `mu`
Nachkommen. **Zufall nur über das übergebene `numpy.random.Generator`.** Der Emitter weiß nichts von Archiv und Labyrinth: er bekommt von außen die Rangfolge der Nachkommen (`tell`) und liefert neue Stichproben (`ask`)."""

import numpy as np


class Emitter:
    """Ein CMA-ES mit `lam` Nachkommen je Generation im `n`-dimensionalen Suchraum."""

    def __init__(self, n, lam, sigma0, rng):
        self.n, self.lam, self.sigma0, self.rng = n, lam, sigma0, rng
        self.chi_n = np.sqrt(n) * (1 - 1 / (4 * n) + 1 / (21 * n * n))
        self.reset(np.zeros(n))

    def reset(self, mean):
        """Neustart bei `mean`: Schrittweite sigma0, Kovarianz Einheitsmatrix, Pfade null."""
        n = self.n
        self.m, self.sigma = np.array(mean, float), self.sigma0
        self.C = np.eye(n)
        self.B, self.D = np.eye(n), np.ones(n)
        self.pc, self.ps = np.zeros(n), np.zeros(n)
        self.gen_since_eig = 0
        self.age = 0
        self.stagnation = []
        self.y = None

    def ask(self):
        """`lam` Stichproben x = m + sigma · B D z mit z ~ N(0, I); merkt sich y = B D z."""
        z = self.rng.standard_normal((self.lam, self.n))
        self.y = (z * self.D) @ self.B.T
        return self.m + self.sigma * self.y

    def condition(self):
        """Konditionszahl der Kovarianzmatrix (größter durch kleinsten Eigenwert); 1 heißt: kugelförmig, die Verteilung hat nichts gelernt."""
        ev = np.linalg.eigvalsh((self.C + self.C.T) / 2)
        return float(ev.max() / max(ev.min(), 1e-300))

    def tell(self, ranked):
        """Aktualisiert Mittelwert, Pfade, Kovarianz und Schrittweite mit den Nachkommen der letzten `ask`-Stichprobe, in der Reihenfolge `ranked` (beste zuerst, Indizes in 0 bis lam − 1). Die Gewichte
        gelten für genau diese `len(ranked)` Nachkommen (log(mu + ½) − log(Rang), auf Summe 1 normiert)."""
        n = self.n
        mu = len(ranked)
        w = np.log(mu + 0.5) - np.log(np.arange(1, mu + 1))
        w = w / w.sum()
        mueff = 1.0 / np.sum(w ** 2)
        cc = (4 + mueff / n) / (n + 4 + 2 * mueff / n)
        cs = (mueff + 2) / (n + mueff + 5)
        c1 = 2 / ((n + 1.3) ** 2 + mueff)
        cmu = min(1 - c1, 2 * (mueff - 2 + 1 / mueff) / ((n + 2) ** 2 + mueff))
        damps = 1 + 2 * max(0.0, np.sqrt((mueff - 1) / (n + 1)) - 1) + cs
        ysel = self.y[list(ranked)]
        yw = w @ ysel
        self.m = self.m + self.sigma * yw
        inv_sqrt_c_yw = self.B @ ((self.B.T @ yw) / self.D)
        self.ps = (1 - cs) * self.ps + np.sqrt(cs * (2 - cs) * mueff) * inv_sqrt_c_yw
        self.age += 1
        self.gen_since_eig += 1
        hsig = float(np.linalg.norm(self.ps) / np.sqrt(1 - (1 - cs) ** (2 * self.age)) / self.chi_n < 1.4 + 2 / (n + 1))
        self.pc = (1 - cc) * self.pc + hsig * np.sqrt(cc * (2 - cc) * mueff) * yw
        self.C = (1 - c1 - cmu) * self.C + c1 * (np.outer(self.pc, self.pc) + (1 - hsig) * cc * (2 - cc) * self.C) + cmu * (ysel.T * w) @ ysel
        self.sigma *= float(np.exp((cs / damps) * (np.linalg.norm(self.ps) / self.chi_n - 1)))
        if self.gen_since_eig >= max(1, int(1 / ((c1 + cmu) * n * 10))):
            self.C = (self.C + self.C.T) / 2
            ev, vec = np.linalg.eigh(self.C)
            self.B, self.D = vec, np.sqrt(np.maximum(ev, 1e-14))
            self.gen_since_eig = 0

    def converged(self, best_value, patience=25, min_step=1e-4, max_condition=1e12):
        """Abbruchkriterium für CMA-MAE: Schrittweite mal größte Achse unter `min_step`, Konditionszahl über `max_condition`, oder in den letzten `patience` Generationen kein besserer bester Wert (`best_value` = bester Wert
        der letzten Generation, größer ist besser) als davor."""
        self.stagnation.append(float(best_value))
        if self.sigma * float(self.D.max()) < min_step or self.condition() > max_condition:
            return True
        return len(self.stagnation) > patience and max(self.stagnation[-patience:]) <= max(self.stagnation[:-patience])
