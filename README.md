# WMOF: Windowing Mass-ratio-variance Outlier Factor

**WMOF** is an anomaly detection algorithm designed for both batch datasets and real-time data streams. It is an extension of the Mass-ratio-variance Outlier Factor (MOF) algorithm that introduces a sliding window mechanism. This allows it to process large datasets efficiently and detect anomalies that occur at the boundaries of data partitions without missing context.

The algorithm is optimized for high performance using Numba JIT compilation, making it suitable for high-throughput environments.

## ✨ Key Features

* **Mass-Ratio Variance Logic:** Detects outliers by analyzing the "mass" density of points relative to their neighbors using rank-ordered distances.
* **Windowing Strategy:** Operates on overlapping fixed-size windows to handle infinite streams and large datasets.
* **High Performance:** Critical mathematical functions (`_point_in_radius`, `_Var_Massratio`) are compiled to machine code using `@jit(nopython=True)` from Numba.
* **Dual Mode:** Supports both `fit()` for static datasets and `fit_score()` for point-by-point streaming.

---

## 📊 Dataset Information

To assess the performance of WMOF, the proposed method is evaluated on 14 diverse datasets: 10 from the UCI Machine Learning Repository, two (mammography and mulcross) from OpenML and two (CATS and creditcardfraud) from Kaggle. The UCI datasets include: ai4i_2020_predictive_maintenance, shuttle, annthyroid, satellite, smtp, cover, pima, breastw, arrhythmia, and ionosphere.

| Dataset | Source | DOI / URL |
| :--- | :--- | :--- |
| AI4I 2020 | UCI | https://doi.org/10.24432/C5HS5C |
| Shuttle | UCI | https://doi.org/10.24432/C5WS31 |
| Annthyroid | UCI | https://doi.org/10.24432/C5D010 |
| Satellite | UCI | https://doi.org/10.24432/C55887 |
| SMTP | UCI | https://doi.org/10.24432/C51C7N |
| Covertype | UCI | https://doi.org/10.24432/C50K5N |
| BreastW | UCI | https://doi.org/10.24432/C5HP4Z |
| Arrhythmia | UCI | https://doi.org/10.24432/C5BS32 |
| Ionosphere | UCI | https://doi.org/10.24432/C5W01B |
| Pima | OpenML | (originally UCI) https://www.openml.org/d/37 |
| Mulcross | OpenML | https://www.openml.org/d/40897 |
| Mammography | OpenML | https://www.openml.org/d/310 |
| CreditCard | Kaggle | https://www.kaggle.com/mlg-ulb/creditcardfraud |
| CATS | Kaggle | https://doi.org/10.5281/zenodo.7646897 |

| Dataset         |   Dimensions |   Total Points |         Anomalies class|   Percentage of Anomalies (%) |
|:----------------|-------------:|---------------:|----------------------:|------------------------------:|
| AI4I_2020       |            6 |          10000 |       Machine failure |                          3.39 |
| Mammography     |            6 |          11183 |               class 1 |                          0.04 |
| Shuttle         |            9 |          49097 |     classes 2,3,5,6,7 |                          0.37 |
| Annthyroid      |            6 |           7200 |         classes 1, 2  |                          0.31 |
| Satellite       |           36 |           6435 |    3 smallest classes |                          3.79 |
| Smtp            |            3 |          95156 |                attack |                          0.03 |
| Cover           |           10 |         286048 |   class 4 vs. class 2 |                          0.07 |
| Mulcross        |            4 |         262144 |            2 clusters |                         10    |
| Pima            |            8 |            768 |                   pos |                          1.56 |
| Breastw         |            9 |            683 |             malignant |                         34.99 |
| Arrhythmia      |          274 |            452 |   classes 3,4,5,7,8,9,14,15 |                         14.6  |
| Ionosphere      |           33 |            351 |                   bad |                         35.9  |
| CATS            |           17 |         300000 |                 11400 |                          3.8  |
| Creditcardfraud |           29 |         284807 |                   492 |                          0.17 |

For the CATS dataset, only the last 300,000 data points were utilized due to resource limitations. For cover dataset, outlier detection dataset is created using only 10 quantitative attributes. instances from class 2 are considered as normal points and instances from class 4 are anomalies. Instances from the other classes are omitted.

## 📚 Code Information

### `WMOF(window=1000, overlap_ratio=0.2)`
* **`window`** (int): Size of the processing window. Default is `1000`.
* **`overlap_ratio`** (float): Ratio of overlap between consecutive windows (must be between `0.0` and `0.5`). Default is `0.2`.

### Methods

#### `fit(data)`
Fits the model on a static dataset (offline mode).
* **`data`**: Numpy array of shape `(n_samples, n_features)`. The input samples.
* **Returns**: `self`.

#### `fit_score(x)`
Ingests a single data point for streaming processing. When the internal buffer reaches `window_size`, it calculates and returns scores for the non-overlapping segment.
* **`x`**: Single data point (array-like) of shape `(1, n_features)`.
* **Returns**: Numpy array of decision scores. Returns an empty array if the window buffer is not yet full.

#### `fit_last_score()`
Calculates scores for any remaining data in the buffer that did not form a complete window (usually called at the end of a stream).
* **Returns**: Numpy array of decision scores for the remaining points.

#### `detectAnomaly(threshold)`
Identifies anomalies in the processed data based on a fixed threshold.
* **`threshold`** (float): The score above which a point is considered an anomaly.
* **Returns**: Numpy array of indices representing the anomaly points.

#### `detectStream(scores, tau=None, n=0.01)`
Helper method to identify anomalies within a specific batch of scores.
* **`scores`**: Numpy array of decision scores to evaluate.
* **`tau`** (float, optional): A manual threshold value.
* **`n`** (float or int, default `0.01`): Used if `tau` is `None`.
    * If `float` ($0.01 \le n \le 0.49$): Selects the top $n\%$ of scores as anomalies.
    * If `int`: Selects the top $n$ number of data points as anomalies.
* **Returns**: Numpy array of indices relative to the input `scores` array.


## 🛠 Requirements

### Core Dependencies
To run the WMOF algorithm, the following libraries are required:

* **NumPy**: Array manipulation.
* **SciPy**: Distance matrix calculations (`cdist`).
* **Numba**: JIT compilation for speed.

```bash
pip install numpy scipy numba
```

### Installation via pymof
Alternatively, you can install the complete package directly from GitHub:
```bash
pip install pymof
```
### Baseline & Dataset Utilities
To use the included baseline models and dataset providers, install these additional dependencies:

* **PySAD**: Framework for anomaly detection baselines.
* **ucimlrepo & kagglehub**: Automated dataset acquisition.
* **mat73**: Datasets reader.

```bash
pip install pysad mat73 ucimlrepo
```

## 🚀 Usage Instructions

1. Data Acquisition

    Fetch from UCI Repository

    ```python
    from ucimlrepo import fetch_ucirepo

    ai4i_2020_predictive_maintenance_dataset = fetch_ucirepo(id=601)

    X = ai4i_2020_predictive_maintenance_dataset.data.features
    y = ai4i_2020_predictive_maintenance_dataset.data.targets
    ```

    Fetch from Kaggle

    ```python
    import kagglehub

    path = kagglehub.dataset_download("mlg-ulb/creditcardfraud")

    sample = pd.read_csv(path)
    df_x = sample.drop(columns=['Time', 'Class'])
    df_y = sample['Class']

    X = df_x.to_numpy().astype("float64")
    y = df_y.to_numpy().astype("int32")
    ```

2. Running WMOF

    The WMOF implementation follows a scikit-learn-like API for ease of use.

    ```python
    from pymof import WMOF
    import numpy as np
    model = WMOF()
    model.fit(X)
    scores = model.decision_scores_
    print(scores)
    ```

## 🔬 Methodology

The experimental framework was designed to evaluate the performance of the proposed WMOF algorithm against standard baseline models in a streaming anomaly detection context. The process consisted of the following steps:

1. **Data Preprocessing**: To ensure feature consistency across varying scales, raw data instances were individually normalized using an Instance Unit Norm Scaler (InstanceUnitNormScaler from pysad). This step was performed online as data arrived, rather than pre-calculating global statistics, to strictly adhere to streaming constraints.

2. **Streaming Simulation**: Real-time data ingestion was simulated using an ArrayStreamer. This utility fed the dataset sequentially to the models (instance-by-instance), preserving the temporal order of the data and preventing look-ahead bias.

3. **Baseline Models**: Standard batch anomaly detection algorithms (IForest, LOF, OCSVM, and KNN) were adapted for the streaming environment using a Reference Window Model (ReferenceWindowModel from pysad).

    * Initialization: The models were initially fitted on a small seed window.

    * Updates: As new data arrived, the models utilized a sliding reference window to update their internal states and score the new instances dynamically.

5. **Proposed Model (WMOF)**: The proposed WMOF algorithm was evaluated using window sizes of 1000 datapoints and a 10% of overlap ratio. Unlike the reference window wrappers used for baselines, WMOF utilized its internal sliding window mechanism (fit_score and fit_last_score) to batch process the stream and calculate mass-ratio variance scores.

6. **Performance Evaluation**: The efficacy of each model was measured using the following metrics:

    * Detection Accuracy: The Area Under the Receiver Operating Characteristic curve (AUROC) were calculated to assess the trade-off between true positives and false positives.

    * Computational Efficiency: The execution time was recorded for each dataset to compare the processing speed of the algorithms.


## Citations 

Blackard, J. (1998). Covertype [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C50K5N

Dal Pozzolo, A., Caelen, O., Johnson, R. A., & Bontempi, G. (2015). Calibrating probability with undersampling for unbalanced classification. Symposium on Computational Intelligence and Data Mining (CIDM), IEEE.

Dua, D., & Graff, C. (1993). Statlog (Shuttle) [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5WS31

Dua, D., & Graff, C. (2020). AI4I 2020 Predictive Maintenance Dataset [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C

Fleith, P. (2023). Controlled Anomalies Time Series (CATS) dataset (Version 2) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.7646897. Retrieved from Kaggle: https://www.kaggle.com/datasets/patrickfleith/controlled-anomalies-time-series-dataset

Guvenir, H., Acar, B., Muderrisoglu, H., & Quinlan, R. (1997). Arrhythmia [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5BS32

Mangasarian, O. L., & Wolberg, W. H. (1990). Cancer diagnosis via linear programming. SIAM News, 23(5), 1–18.

OpenML. (n.d.). Mulcross (ID 40897) [Data set]. https://www.openml.org/d/40897

OpenML. (n.d.). mammography (ID 310) [Data set]. https://www.openml.org/d/310

OpenML. (n.d.). Pima Indians Diabetes Database (ID 37) [Data set]. https://www.openml.org/d/37

Quinlan, R. (1986). Thyroid Disease [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5D010

Rocke, D. M., & Woodruff, D. L. (1996). Identification of outliers in multivariate data. Journal of the American Statistical Association, 91(435), 1047–1061.

Sigillito, V., Wing, S., Hutton, L., & Baker, K. (1989). Ionosphere [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5W01B

Srinivasan, A. (1993). Statlog (Landsat Satellite) [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C55887

Stolfo, S., Fan, W., Lee, W., Prodromidis, A., & Chan, P. (1999). KDD Cup 1999 Data [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C51C7N

ULB Machine Learning Group. (2016). Credit Card Fraud Detection [Data set]. Kaggle. https://www.kaggle.com/mlg-ulb/creditcardfraud

Wolberg, W. (1990). Breast Cancer Wisconsin (Original) [Data set]. UCI Machine Learning Repository. https://doi.org/10.24432/C5HP4Z

Woods, K. S., Doss, C. C., Bowyer, K. W., Solka, J. L., Priebe, C. E., & Kegelmeyer Jr, W. P. (1993). Comparative evaluation of pattern recognition techniques for detection of microcalcifications in mammography. International Journal of Pattern Recognition and Artificial Intelligence, 7(06), 1417–1436.

Yamanishi, K., Takeuchi, J. I., Williams, G., & Milne, P. (2000). Online unsupervised outlier detection using finite mixtures with discounting learning algorithms. Proceedings of the Sixth ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 320–324.

## License & Contribution Guidelines

MIT License

Copyright (c) 2025 Supakit

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.