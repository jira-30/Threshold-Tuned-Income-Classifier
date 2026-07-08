# Data

This folder holds the **UCI Adult Census Income** dataset.

The notebook downloads the data automatically on first run via
`sklearn.datasets.fetch_openml('adult', version=2, as_frame=True)` and caches
it locally. You do not need to download anything by hand.

If you are running offline, you can also fetch the original files from the
UCI ML Repository and place them in this folder:

- https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data
- https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.test
- https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.names

**Citation:**
Becker, B. & Kohavi, R. (1996). Adult [Dataset]. UCI Machine Learning
Repository. https://doi.org/10.24432/C5XW20
