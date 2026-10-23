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
| `species` | string | — | One of `setosa`, `versicolor`, `virginica` |

## Class balance

| Species | Count |
| --- | ---: |
| setosa | 50 |
| versicolor | 50 |
| virginica | 50 |

There are no missing values. The file is generated reproducibly with
`python src/run_all.py --refresh-data`, which writes the scikit-learn copy to
this path. The original dataset is available at
<https://archive.ics.uci.edu/dataset/53/iris>.

## Citation

Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems.
*Annals of Eugenics*, 7(2), 179–188.
