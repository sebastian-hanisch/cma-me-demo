# CMA-ME und CMA-MAE – wann lohnt es sich, die Suchverteilung zu lernen? (Streamlit-Demo)

---

Interaktive Demo zu **CMA-ME und CMA-MAE**. **Sechstes Stück der Konzepte-Linie „Novelty Search und Quality-Diversity“** im Portfolio von
[Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning): ein Verfahren, ein wachsendes Beispiel, jedes Folgestück hebt genau eine Annahme auf.

Derselbe Hof-Roboter, dasselbe Archiv (Endposition in Blöcken der Kantenlänge 2, 144 Archivzellen) und dieselbe Qualität (Direktheit der Fahrt) wie in [`map-elites-demo`](https://github.com/sebastian-hanisch/map-elites-demo) (Stück 5). MAP-Elites erzeugt Kinder durch feste Mutation
eines zufälligen Elites. **CMA-ME** gibt jeder Teilsuche (einem **Emitter**) ein **CMA-ES**, das die Richtung und Stärke der Verbesserungen lernt; **CMA-MAE** ergänzt je Zelle eine **Schwelle**, die mit einer Lernrate α langsam zur besten Qualität aufsteigt. Das Stück misst, **wann das gegenüber
MAP-Elites etwas bringt**: bei kurzen Steuerfolgen (wenige Dimensionen) deutlich, bei den 280 Dimensionen der Fahrten aus den früheren Stücken nicht, auch nicht in den Labyrinthen der Stufen 3 und 4.

## Kernfrage

Ab welcher Dimension des Suchraums verliert das Lernen der Suchverteilung seinen Vorteil gegenüber fester Mutation, und wie empfindlich sind die Verfahren gegen ihre Einstellungen?

## Modell und Methodik

- **Labyrinth, Roboter, Archiv** (`cme_maze.py`, `cme_robot.py`, `cme_search.py`): wie Stück 1 bis 5; Stufe 3 und 4 bei 140 Schritten, dazu das **freie Feld** (Stufe 0) mit Steuerfolgen von 10 bis 140 Schritten: die Dimension des Suchraums ist 2 × Länge (Drehung und Tempo je
  Schritt), also 20 bis 280. **Qualität = Direktheit:** Luftlinie von Start zu Ende geteilt durch die gefahrene Weglänge (0 bis 1). Das **Ergebnisarchiv** hält je Zelle die beste Fahrt (ein Kind ersetzt den Elite nur, wenn es um mehr als 10⁻¹² besser ist); alle Kennzahlen
  kommen daraus und sind bei allen Verfahren gleich definiert: **QD-Score** (Summe der Direktheit über die belegten Zellen, eigene Umsetzung der in der Quality-Diversity-Literatur üblichen Summenkennzahl), Abdeckung, beste Direktheit.
- **MAP-Elites** (Bezug, wie Stück 5): 100 Kinder je Generation aus gleichverteilt gezogenen Elites, 15 % der Gene mit Gauß 0.25 verrauscht.
- **Emitter** (`cme_cmaes.py`): 5 Emitter mit je 20 Kindern je Generation (100 Evaluationen wie bei MAP-Elites). Jeder hält ein **CMA-ES** nach dem Tutorial von Hansen (2016): Mittelwert, Schrittweite σ, volle Kovarianzmatrix (träge Eigenzerlegung), Evolutionspfade, logarithmische Gewichte.
  Start bei einem zufälligen Elite mit Schrittweite σ₀.
- **CMA-ME** (Verbesserungs-Emitter, Fontaine u. a. 2020): jedes Kind wird ins Archiv eingetragen; es heißt *neue Zelle* (die Zelle war leer) oder *Verbesserung* (Qualität um mehr als 10⁻¹² über dem Elite). Der Emitter ordnet nur eingetragene Kinder (erst neue Zellen nach
  Qualität, dann Verbesserungen nach Größe) und lernt aus ihnen; trug in einer Generation kein Kind ein, startet er bei einem zufälligen Elite neu.
- **CMA-MAE** (Fontaine und Nikolaidis 2023, Lernrate α = 0.01 / 0.1 / 0.5): je Zelle eine **Schwelle** (Start 0). Verbesserung Δ = Qualität − Schwelle; ist die Qualität größer als die Schwelle, ersetzt das Kind den Insassen des Emitter-Archivs und die Schwelle steigt auf
  (1 − α) · Schwelle + α · Qualität. Der Emitter ordnet **alle** Kinder nach Δ und lernt aus der besseren Hälfte; er startet neu, wenn sein CMA-ES konvergiert ist (Schrittweite mal größte Achse unter 10⁻⁴, Konditionszahl über 10¹² oder 25 Generationen ohne besseren besten Wert).
- **Eigene Umsetzung:** die Emitter sind eine eigene NumPy-Umsetzung nach den Veröffentlichungen, keine Referenzbibliothek. Die Neustart-Regel von CMA-MAE (Konvergenz) ist mit eigenen, hier festgelegten Schwellen umgesetzt.
- **Gegenproben:** (1) das CMA-ES-Update von Hand an einem 2D-Beispiel gegen eine skalare Neuberechnung in reinen Python-Listen nach Hansens Formeln; (2) **pycma** als unabhängige Referenz: mit denselben Stichproben stimmt der Mittelwert auf 10⁻¹² überein, Schrittweite und Kovarianz
  bleiben über 6 Schritte innerhalb von 3 % und 0.05 (pycma weicht in zwei Lernraten vom Tutorial ab: die Pfadlänge cs hat dort den Nenner N + μeff + 3 statt + 5, und die Rang-μ-Rate cμ ist anders; sie werden für den Vergleich auf die Tutorial-Werte gesetzt); die Zahl der
  Evaluationen bis zur Konvergenz auf einem Ellipsoid mit Konditionszahl 10⁶ liegt zwischen dem 0.5- und 2-fachen von pycma; (3) Archiv und Emitter-Regeln von Hand (neue Zelle vor Verbesserung, Rangfolge, Schwellen mit Lernrate: Δ, Anstieg, kein Eintrag bei Δ ≤ 0, α = 1 gleich dem besten
  Wert, ein Insasse kann schlechter sein als der Ergebnis-Elite); (4) **Orakel:** MAP-Elites, CMA-ME und CMA-MAE **als Ganzes** gegen eine skalare Neuimplementierung der Buchführung mit Wörterbüchern (gleiche Zufallszahlen, am Ende dieselben Zellen, Qualitäten, Steuerfolgen und Kurven);
  die Direktheit gegen eine skalare Neuberechnung der Fahrt bei Genomlängen von 5 bis 140; (5) kein Verfahren benutzt das Ziel: verschobenes Ziel ändert die Archiv-Kurven nicht; (6) MAP-Elites reproduziert die Zahlen aus Stück 5; (7) Studienläufe von MAP-Elites werden mit demselben Seed bitgleich reproduziert, die der CMA-Verfahren fallen als frischer Lauf in die gespeicherte Verteilung (siehe Grenzen).
- **Vorgerechnete Studie** (`generate_precomputed.py` → `precomputed_sweep.json`, rund 20 Minuten parallel): fünf Verfahren (MAP-Elites, CMA-ME, CMA-MAE mit α = 0.01 / 0.1 / 0.5) in sieben Fällen (freies Feld mit 10 / 20 / 40 / 80 / 140 Schritten, Stufe 3, Stufe 4), σ₀ = 0.2, je 20 Läufe
  à 400 Generationen (40 000 Evaluationen); dazu σ₀ = 0.05 / 0.1 / 0.2 / 0.4 für CMA-ME und CMA-MAE (α = 0.1) im freien Feld bei 40 und 140 Schritten.

## Befunde (gemessen, keine Behauptungen)

Alle Zahlen stehen in `tests/test_claims.py`. 20 Läufe je Zelle (Seeds 0 bis 19), Zellgröße 2 (144 Archivzellen), σ₀ = 0.2; „±“ ist die Standardabweichung über die Läufe, „p“ der Mann-Whitney-Test des QD-Scores gegen MAP-Elites.

| Fall | Verfahren | QD-Score (Mittel ± SD) | Gewinn gegenüber MAP-Elites | p | belegte Zellen (von 144) | beste Direktheit | Neustarts je Lauf | Erfolg | Restweg |
|---|---|---|---|---|---|---|---|---|---|
| freies Feld, 10 Schritte (20 Dimensionen) | MAP-Elites (Bezug aus Stück 5) | 20.30 ± 2.66 | – | – | 20.30 | 1.0000 | 0 | 0 von 20 | 11.85 |
| freies Feld, 10 Schritte (20 Dimensionen) | CMA-ME | 23.28 ± 1.12 | +14.7 % | < 0.001 | 23.30 | 1.0000 | 1809 | 0 von 20 | 10.85 |
| freies Feld, 10 Schritte (20 Dimensionen) | CMA-MAE, Lernrate 0.01 | 30.75 ± 1.02 | +51.5 % | < 0.001 | 30.80 | 1.0000 | 53 | 0 von 20 | 10.00 |
| freies Feld, 10 Schritte (20 Dimensionen) | CMA-MAE, Lernrate 0.1 | 28.92 ± 2.40 | +42.5 % | < 0.001 | 28.95 | 1.0000 | 47 | 0 von 20 | 10.30 |
| freies Feld, 10 Schritte (20 Dimensionen) | CMA-MAE, Lernrate 0.5 | 25.27 ± 2.49 | +24.5 % | < 0.001 | 25.30 | 1.0000 | 43 | 0 von 20 | 10.90 |
| freies Feld, 20 Schritte (40 Dimensionen) | MAP-Elites (Bezug aus Stück 5) | 51.90 ± 5.09 | – | – | 52.15 | 1.0000 | 0 | 0 von 20 | 5.95 |
| freies Feld, 20 Schritte (40 Dimensionen) | CMA-ME | 58.21 ± 2.66 | +12.2 % | < 0.001 | 58.75 | 0.9996 | 1619 | 0 von 20 | 4.95 |
| freies Feld, 20 Schritte (40 Dimensionen) | CMA-MAE, Lernrate 0.01 | 71.84 ± 0.65 | +38.4 % | < 0.001 | 72.25 | 1.0000 | 48 | 0 von 20 | 3.10 |
| freies Feld, 20 Schritte (40 Dimensionen) | CMA-MAE, Lernrate 0.1 | 70.67 ± 1.59 | +36.2 % | < 0.001 | 70.90 | 1.0000 | 45 | 0 von 20 | 3.35 |
| freies Feld, 20 Schritte (40 Dimensionen) | CMA-MAE, Lernrate 0.5 | 66.72 ± 4.11 | +28.6 % | < 0.001 | 67.00 | 1.0000 | 41 | 0 von 20 | 4.30 |
| freies Feld, 40 Schritte (80 Dimensionen) | MAP-Elites (Bezug aus Stück 5) | 127.34 ± 2.92 | – | – | 131.65 | 0.9992 | 0 | 20 von 20 | 0.00 |
| freies Feld, 40 Schritte (80 Dimensionen) | CMA-ME | 128.69 ± 3.05 | +1.1 % | 0.190 | 133.15 | 0.9970 | 1378 | 20 von 20 | 0.00 |
| freies Feld, 40 Schritte (80 Dimensionen) | CMA-MAE, Lernrate 0.01 | 138.78 ± 1.24 | +9.0 % | < 0.001 | 142.40 | 0.9990 | 47 | 20 von 20 | 0.00 |
| freies Feld, 40 Schritte (80 Dimensionen) | CMA-MAE, Lernrate 0.1 | 138.95 ± 1.28 | +9.1 % | < 0.001 | 141.35 | 0.9997 | 45 | 20 von 20 | 0.00 |
| freies Feld, 40 Schritte (80 Dimensionen) | CMA-MAE, Lernrate 0.5 | 133.76 ± 2.42 | +5.0 % | < 0.001 | 137.00 | 0.9998 | 42 | 20 von 20 | 0.00 |
| freies Feld, 80 Schritte (160 Dimensionen) | MAP-Elites (Bezug aus Stück 5) | 133.91 ± 0.87 | – | – | 144.00 | 0.9971 | 0 | 20 von 20 | 0.00 |
| freies Feld, 80 Schritte (160 Dimensionen) | CMA-ME | 132.78 ± 1.23 | −0.8 % | 0.005 | 144.00 | 0.9937 | 1342 | 20 von 20 | 0.00 |
| freies Feld, 80 Schritte (160 Dimensionen) | CMA-MAE, Lernrate 0.01 | 136.22 ± 1.57 | +1.7 % | < 0.001 | 144.00 | 0.9946 | 46 | 20 von 20 | 0.00 |
| freies Feld, 80 Schritte (160 Dimensionen) | CMA-MAE, Lernrate 0.1 | 138.21 ± 1.00 | +3.2 % | < 0.001 | 143.95 | 0.9986 | 45 | 20 von 20 | 0.00 |
| freies Feld, 80 Schritte (160 Dimensionen) | CMA-MAE, Lernrate 0.5 | 136.77 ± 1.76 | +2.1 % | < 0.001 | 144.00 | 0.9995 | 43 | 20 von 20 | 0.00 |
| freies Feld, 140 Schritte (280 Dimensionen) | MAP-Elites (Bezug aus Stück 5) | 121.76 ± 0.58 | – | – | 144.00 | 0.9925 | 0 | 20 von 20 | 0.00 |
| freies Feld, 140 Schritte (280 Dimensionen) | CMA-ME | 120.20 ± 0.53 | −1.3 % | < 0.001 | 144.00 | 0.9898 | 1323 | 20 von 20 | 0.00 |
| freies Feld, 140 Schritte (280 Dimensionen) | CMA-MAE, Lernrate 0.01 | 117.86 ± 1.52 | −3.2 % | < 0.001 | 144.00 | 0.9855 | 47 | 20 von 20 | 0.00 |
| freies Feld, 140 Schritte (280 Dimensionen) | CMA-MAE, Lernrate 0.1 | 121.18 ± 0.28 | −0.5 % | < 0.001 | 144.00 | 0.9949 | 48 | 20 von 20 | 0.00 |
| freies Feld, 140 Schritte (280 Dimensionen) | CMA-MAE, Lernrate 0.5 | 120.82 ± 0.62 | −0.8 % | < 0.001 | 144.00 | 0.9963 | 47 | 20 von 20 | 0.00 |
| Stufe 3 (140 Schritte) | MAP-Elites (Bezug aus Stück 5) | 102.96 ± 0.65 | – | – | 144.00 | 0.9932 | 0 | 20 von 20 | 0.00 |
| Stufe 3 (140 Schritte) | CMA-ME | 101.08 ± 0.53 | −1.8 % | < 0.001 | 144.00 | 0.9887 | 1384 | 20 von 20 | 0.00 |
| Stufe 3 (140 Schritte) | CMA-MAE, Lernrate 0.01 | 97.88 ± 2.39 | −4.9 % | < 0.001 | 143.50 | 0.9810 | 46 | 20 von 20 | 0.05 |
| Stufe 3 (140 Schritte) | CMA-MAE, Lernrate 0.1 | 100.75 ± 0.65 | −2.2 % | < 0.001 | 144.00 | 0.9914 | 49 | 20 von 20 | 0.00 |
| Stufe 3 (140 Schritte) | CMA-MAE, Lernrate 0.5 | 100.51 ± 0.71 | −2.4 % | < 0.001 | 143.90 | 0.9964 | 48 | 20 von 20 | 0.10 |
| Stufe 4 (140 Schritte) | MAP-Elites (Bezug aus Stück 5) | 59.29 ± 1.88 | – | – | 85.90 | 0.9972 | 0 | 0 von 20 | 29.75 |
| Stufe 4 (140 Schritte) | CMA-ME | 58.44 ± 1.10 | −1.4 % | 0.047 | 85.70 | 0.9930 | 1571 | 0 von 20 | 29.95 |
| Stufe 4 (140 Schritte) | CMA-MAE, Lernrate 0.01 | 56.65 ± 1.33 | −4.4 % | < 0.001 | 82.80 | 0.9924 | 47 | 0 von 20 | 32.45 |
| Stufe 4 (140 Schritte) | CMA-MAE, Lernrate 0.1 | 57.58 ± 1.04 | −2.9 % | < 0.001 | 82.50 | 0.9973 | 49 | 0 von 20 | 32.55 |
| Stufe 4 (140 Schritte) | CMA-MAE, Lernrate 0.5 | 56.51 ± 1.47 | −4.7 % | < 0.001 | 79.90 | 0.9979 | 47 | 0 von 20 | 33.75 |

| Frage | Befund |
|---|---|
| Wann gewinnt CMA gegenüber MAP-Elites? | **Bei wenigen Dimensionen, deutlich.** Gewinn des QD-Scores bei 20 / 40 / 80 / 160 / 280 Dimensionen: CMA-ME +14.7 / +12.2 / +1.1 / −0.8 / −1.3 %; CMA-MAE mit Lernrate 0.01: +51.5 / +38.4 / +9.0 / +1.7 / −3.2 %; mit 0.1: **+42.5 / +36.2 / +9.1 / +3.2 / −0.5 %**; mit 0.5: +24.5 / +28.6 / +5.0 / +2.1 / −0.8 %. Alle Gewinne bei 20 und 40 Dimensionen haben p < 0.001; CMA-ME bei 80 Dimensionen (+1.1 %, p = 0.19) ist nicht belegt. Ab 40 Dimensionen sinkt der Gewinn mit jeder Verdopplung. |
| Und bei der Dimension der früheren Stücke (280)? | **Kein Vorsprung, leicht darunter.** Alle vier CMA-Verfahren liegen im freien Feld mit 140 Schritten zwischen −0.5 % und −3.2 % unter MAP-Elites (jeweils p < 0.001, aber klein), auf Stufe 3 zwischen −1.8 % und −4.9 %, auf Stufe 4 zwischen −1.4 % (CMA-ME, p = 0.047, ein Grenzfall) und −4.7 %. Stufe 3 lösen weiter alle Verfahren in 20 von 20 Läufen, Stufe 4 keines (0 von 20). |
| Woher kommt der Gewinn bei wenigen Dimensionen? | **Vor allem aus mehr belegten Zellen.** Bei 20 Schritten belegt MAP-Elites 52.15 von 144 Zellen, CMA-ME 58.75, CMA-MAE (0.01 / 0.1 / 0.5) 72.25 / 70.90 / 67.00; ab 80 Schritten belegen alle Verfahren (fast) alle 144 Zellen, dort geht es nur um die Qualität je Zelle. Die Streuung ist bei CMA-MAE viel kleiner als bei MAP-Elites (Standardabweichung 0.65 gegen 5.09 bei 20 Schritten mit Lernrate 0.01, 1.59 mit 0.1). |
| Welche Lernrate ist die beste? | Bei wenigen Dimensionen die kleine (0.01 und 0.1 liegen bei 40 Dimensionen mit 71.84 und 70.67 nah beieinander, 0.5 mit 66.72 darunter); bei 280 Dimensionen ist keine besser als MAP-Elites. Die Lernrate 0.01 ist dort die schlechteste (−3.2 % im freien Feld, −4.9 % auf Stufe 3). |
| Lernt die Kovarianz etwas? | **Wenig, und mit der Dimension weniger.** Die CMA-ME-Emitter starten im Mittel 1 323 bis 1 809 Mal je Lauf neu (von 1 995 Emitter-Generationen) und behalten eine kugelförmige Kovarianz: Zehnerlogarithmus der Konditionszahl 0.00 nach 400 Generationen in jedem Fall. CMA-MAE-Emitter starten 41 bis 53 Mal neu; ihre Kovarianz (Lernrate 0.1) erreicht Zehnerlogarithmen von 0.44 / 0.21 / 0.10 / 0.06 / 0.02 bei 20 / 40 / 80 / 160 / 280 Dimensionen, also Konditionszahlen von 2.8 bis 1.05. |
| Wie wichtig ist σ₀? | **Sehr, für CMA.** Freies Feld mit 40 Schritten, σ₀ = 0.05 / 0.1 / 0.2 / 0.4: CMA-ME 97.92 / 127.28 / 128.69 / 123.28; CMA-MAE (0.1) 126.65 / 136.00 / 138.95 / 136.88; MAP-Elites 127.34 (kennt kein σ₀). Mit 140 Schritten: CMA-ME 117.71 / 121.96 / 120.20 / 114.28, CMA-MAE 108.84 / 120.58 / 121.18 / 117.06, MAP-Elites 121.76. Mit der besten Schrittweite (0.1) holt CMA-ME bei 280 Dimensionen MAP-Elites ein (121.96 gegen 121.76, +0.2 %), aber nicht belegt (p > 0.05). |

## Befunde und Korrekturen gegenüber der Vorab-Messreihe

- **Die erste Fassung war nicht die von CMA-MAE.** Die Vorab-Messreihe setzte den Elite nur bei größerer Qualität als der Elite ersetzt und startete Emitter neu, wenn kein Kind das Archiv verbesserte; das ist CMA-ME. Nach dem Nachlesen der Quelle
  (Fontaine und Nikolaidis 2023) ersetzt ein Kind bei CMA-MAE den Insassen schon, wenn es die **Schwelle** übertrifft, alle Kinder werden nach Δ geordnet, und der Neustart erfolgt bei Konvergenz des CMA-ES. Die gebaute Fassung folgt dem; gemessen wird in einem getrennten
  Ergebnisarchiv (bestes Kind je Zelle), weil ein Insasse schlechter sein kann als der beste bisher gesehene. Die Zahlen blieben danach ähnlich (CMA-MAE 0.1 bei 40 Dimensionen: +37.6 % in der Vorab-Messung mit 8 Läufen, +36.2 % in der Studie mit 20).
- **Ein Fehler in der Vorab-Fassung des Emitters:** die Prüfung der Pfadlänge (`hsig`) benutzte einen Zähler, der bei jeder Eigenzerlegung zurückgesetzt wurde; im gebauten Emitter zählt das Alter des Emitters. Die Vorab-Konditionszahlen (bis 28 bei 20 Dimensionen) waren deutlich höher als die jetzt gemessenen (2.8 oder weniger); wie viel davon auf diesen Fehler und wie viel auf die geänderte Neustart-Regel von CMA-MAE zurückgeht, wurde nicht getrennt untersucht.
- **Der Vorteil hängt an der Dimension, nicht am Labyrinth.** Erwartet war ein Vorsprung auch im Labyrinth. Gemessen ist er dort (280 Dimensionen) leicht negativ, und derselbe Roboter mit kurzer Steuerfolge zeigt den Gewinn.
- **Der Gewinn bei wenigen Dimensionen ist zum großen Teil ein Abdeckungsgewinn** (mehr Zellen), nicht nur bessere Qualität je Zelle; das Archiv füllt sich schneller. Bei 20 Schritten sind nicht alle 144 Zellen erreichbar (der Roboter fährt höchstens 14 Zellen weit): der Vergleich gilt dem Erschließen
  der erreichbaren Zellen.

## Ehrliche Grenzen

- **Eigene Umsetzung, nicht die Referenz.** Einzelheiten (Neustart-Regel von CMA-MAE, Gewichte, Ordnung von Gleichständen) können von der Referenzimplementierung abweichen; das CMA-ES selbst ist gegen pycma geprüft (Mittelwert exakt, Schrittweite und Kovarianz innerhalb von 3 %),
  die Archiv-Buchführung gegen eine unabhängige Neuimplementierung, die Emitter-Logik als Ganzes nicht gegen die Referenzbibliothek von CMA-ME und CMA-MAE.
- **Eine Qualität, eine Zellgröße, ein Budget.** Direktheit, Zellgröße 2 und 40 000 Evaluationen sind fest; mehr Kinder je Emitter oder ein längeres Budget (die Kovarianz könnte in 280 Dimensionen später lernen) wurden nicht gemessen.
- **Die Dimension variiert über die Länge der Steuerfolge.** Das ändert auch, was der Roboter erreichen kann (bei 10 und 20 Schritten sind nicht alle Zellen erreichbar). Eine Dimension ohne Nebenwirkung (etwa zusätzliche wirkungslose Gene) wurde nicht gemessen.
- **Eine kleine Zahl von Emittern und fester σ₀ = 0.2** in der Fallstudie; die σ₀-Wahl ändert die Zahlen deutlich (siehe oben), andere Emitterzahlen sind nicht gemessen.
- **Kleine, aber „signifikante“ Unterschiede.** Bei 20 Läufen und kleiner Streuung sind Unterschiede von 1 % bis 2 % mit p < 0.001 belegt, aber praktisch ohne Bedeutung; die Aussage „kein Vorsprung bei 280 Dimensionen“ meint praktisch gleich, nicht statistisch identisch.
- **CMA-Läufe sind nicht plattformübergreifend bitgleich.** Die Eigenzerlegung (LAPACK) liefert je Plattform andere Eigenvektoren (Vorzeichen, Drehung bei fast gleichen Eigenwerten); ein Emitter mit kugelförmiger Kovarianz zieht dann andere Stichproben. Ein frischer Lauf mit demselben Seed ist deshalb unter Linux ein anderer Zug aus derselben Verteilung als der gespeicherte Windows-Lauf (die Tests prüfen das als Band von 4 Standardabweichungen); MAP-Elites ist bitgleich. Die Mittelwerte der Studie sind davon nicht betroffen (20 Läufe je Zelle), einzelne Läufe wohl.
- **Keine Ziele, kein Labyrinthvergleich:** keines der Verfahren benutzt das Ziel; ob ein Archiv-Verfahren die Schlangenlinie löst, ist Gegenstand der früheren Stücke (Stufe 4: kein Verfahren bei 140 Schritten).

## Verwandte Demos im Portfolio

- [`deception-maze-demo`](https://github.com/sebastian-hanisch/deception-maze-demo): Stück 1, das täuschende Labyrinth.
- [`novelty-search-demo`](https://github.com/sebastian-hanisch/novelty-search-demo): Stück 2, Novelty Search mit und ohne Archiv.
- [`behavior-space-demo`](https://github.com/sebastian-hanisch/behavior-space-demo): Stück 3, zwölf Verhaltensräume.
- [`novelty-objective-demo`](https://github.com/sebastian-hanisch/novelty-objective-demo): Stück 4, Novelty plus Ziel.
- [`map-elites-demo`](https://github.com/sebastian-hanisch/map-elites-demo): Stück 5, MAP-Elites (der Bezug dieses Stücks).
- [`cma-es-demo`](https://github.com/sebastian-hanisch/cma-es-demo): das CMA-ES allein, ohne Archiv.

## Bewusst nicht umgesetzt

Jede dieser Annahmen hebt ein Folgestück der Linie auf:

| Annahme | Folgestück |
|---|---|
| Die Suchverteilung lässt sich lernen | kein Folgestück |
| Die Qualität gibt eine Richtung vor | kein Folgestück |
| Ein festes Raster genügt | CVT-MAP-Elites (Folgestück) |
| Eigene Umsetzung = Referenz | kein Folgestück |
| Eine Suche mit fester Steuerfolge (Stufe 4 hängt am Suchraum) | Go-Explore und POET (Folgestücke) |

## Tests

235 Tests, rund eine Minute: Labyrinth und Roboter (aus Stück 1, Genomlänge unabhängig von der globalen Konstanten), das CMA-ES (Update von Hand und skalar, gegen pycma, Konvergenz auf Kugel und Ellipsoid, Neustart, Konditionszahl, Abbruchkriterium), Archiv und Emitter-Regeln von Hand,
Suche (Reproduzierbarkeit je Verfahren, Kurven, Archiv in sich stimmig, Konditionszahlen zu den Meilensteinen, wer das Ziel sieht), **Orakel** (Direktheit skalar, alle fünf Verfahren als Ganzes gegen eine Wörterbuch-Buchführung, Restweg gegen SciPy), Auswertung und Vollständigkeit der
vorgerechneten Datei (inklusive Neurechnung einzelner Läufe), Presets und Permalink, Diagramme (gesperrte Achsen, Fehlerbalken), AppTest-Rauchtests mit festem Würfel-Seed, der Smoke-Test der Portfolio-Vorlage, ein Quelltext-Test gegen Satz-Komma-Fehler und `test_claims.py` für jede Zahl dieser README.

## Dateistruktur

| Datei | Inhalt |
|---|---|
| `app.py` | Streamlit-Oberfläche |
| `cme_maze.py`, `cme_robot.py` | Labyrinth, Breitensuche, Täuschungsmaße; Kinematik (Genomlänge frei), Population, Pfad-Regler (wie Stück 1) |
| `cme_cmaes.py` | CMA-ES-Emitter (Mittelwert, Schrittweite, Kovarianz, Pfade, Konvergenzkriterium) |
| `cme_search.py` | Direktheit, Archiv, MAP-Elites, CMA-ME, CMA-MAE, Kennzahlen |
| `cme_evaluation.py` | Maße je Stufe, Lösbarkeit, Live-Paar, Zugriff auf die Studie, Mann-Whitney-Hilfe |
| `generate_precomputed.py` | rechnet die Studie vor → `precomputed_sweep.json` |
| `cme_visualization.py` | Plotly-Abbildungen (Achsen gesperrt) |
| `cme_presets.py`, `cme_constants.py` | Presets, Permalink, Grenzen |
| `tests/` | siehe oben |

## Literatur

- Fontaine, M. C., Togelius, J., Nikolaidis, S., Hoover, A. K. (2020): Covariance matrix adaptation for the rapid illumination of behavior space. *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '20)*, 94–102.
- Fontaine, M. C., Nikolaidis, S. (2023): Covariance matrix adaptation MAP-annealing. *Proceedings of the Genetic and Evolutionary Computation Conference (GECCO '23)*; arXiv:2205.10752.
- Hansen, N. (2016): The CMA evolution strategy: a tutorial. arXiv:1604.00772.
- Mouret, J.-B., Clune, J. (2015): Illuminating search spaces by mapping elites. arXiv:1504.04909.

## Lokal ausführen

```
pip install -r requirements.txt
streamlit run app.py
```

Tests: `pip install -r requirements-dev.txt` und `python -m pytest tests/ -v`. Studie neu rechnen: `python generate_precomputed.py` (rund 20 Minuten).

Gebaut mit Streamlit und Plotly (die Simulation und die Emitter rechnen NumPy).
