# Iris dataset
`iris.csv` is the classic Iris flower dataset, obtained from scikit-learn's
built-in copy of the data published by Ronald Fisher (1936) and now hosted in
the UCI Machine Learning Repository. It contains 150 observations, 50 per
species, and four continuous measurements in centimetres.

## Data dictionary

| Column | Type | Unit | Description |
| --- | --- | --- | --- |
| `sepal length (cm)` | float | cm | Length of the sepal |
| `sepal width (cm)` | float | cm | Width of the sepal |
| `petal length (cm)` | float | cm | Length of the petal |
| `petal width (cm)` | float | cm | Width of the petal |
| `species` | string | — | One of `Iris setosa`, `Iris versicolor`, `Iris virginica` |

## Class balance

| Species | Count |
| --- | ---: |
| Iris setosa | 50 |
| Iris versicolor | 50 |
| Iris virginica | 50 |

There are no missing values. The file is generated reproducibly with
`python src/run_all.py --refresh-data`, which writes the scikit-learn copy to
this path. The original dataset is available at
<https://archive.ics.uci.edu/dataset/53/iris>.

## Citation

Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems.
*Annals of Eugenics*, 7(2), 179–188.

# AI4I 2020 Predictive Maintenance dataset
`ai4i2020.csv` is the UCI AI4I 2020 Predictive Maintenance dataset
(UCI ML Repository #601, DOI 10.24432/C5HS5C). It is a synthetic
representation of a milling process used in the IEEE AI4I 2020
challenge and contains 10,000 observations with no missing values.

## Data dictionary

| Column | Type | Unit | Role | Description |
| --- | --- | --- | --- | --- |
| `UDI` | int | — | identifier | Unique data-point identifier (row order) |
| `Product ID` | string | — | identifier | Alphanumeric product batch code |
| `Type` | string | — | feature | Product size class: `L`, `M`, `H` |
| `Air temperature [K]` | float | K | feature | Air-temperature measurement |
| `Process temperature [K]` | float | K | feature | Process-temperature measurement |
| `Rotational speed [rpm]` | int | rpm | feature | Spindle speed |
| `Torque [Nm]` | float | N·m | feature | Torque measurement |
| `Tool wear [min]` | int | min | feature | Accumulated tool-wear time |
| `Machine failure` | int | — | target | Binary: 1 when the machine failed |
| `TWF` | int | — | target | Tool-wear-failure flag |
| `HDF` | int | — | target | Hydrodynamic-force-failure flag |
| `PWF` | int | — | target | Power-derating-failure flag |
| `OSF` | int | — | target | Over-speed-failure flag |
| `RNF` | int | — | target | Run-out-failure flag |

## Class balance

| Class | Count | Share |
| --- | ---: | ---: |
| No failure (`Machine failure = 0`) | 9,661 | 96.61% |
| Machine failure (`Machine failure = 1`) | 339 | 3.39% |

Per-mode counts (some rows carry more than one active flag):

| Mode | Count |
| --- | ---: |
| TWF | 46 |
| HDF | 115 |
| PWF | 95 |
| OSF | 98 |
| RNF | 19 |

There are no missing values. The file is committed directly to the
repository; refresh it with `python src/ai4i_run.py` (downloads from
`https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv`
when the local copy is absent).

## Citation

Matzka, S. (2020). Explainable Artificial Intelligence for Predictive
Maintenance Applications. In *Proceedings of the Third International
Conference on Artificial Intelligence for Industries (AI4I)*, pp. 69–74.
IEEE. DOI 10.1109/AI4I49448.2020.00023.

UCI Machine Learning Repository. AI4I 2020 Predictive Maintenance
Dataset. DOI 10.24432/C5HS5C.
