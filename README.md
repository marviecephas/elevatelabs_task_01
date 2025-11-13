# Task 1: Data Cleaning and Preprocessing (Titanic Dataset)

### Objective
To prepare the raw Titanic dataset for Machine Learning model training by executing a comprehensive data cleaning and feature engineering pipeline. The goal was to ensure data quality, handle missingness, and transform categorical features into a standardized numerical format.

### Tools Used
* **Python**
* **Pandas** (Data manipulation and cleaning)
* **NumPy** (Numerical operations)
* **Seaborn/Matplotlib** (Visualization for outliers)
* **Scikit-learn (StandardScaler, LabelEncoder)** (Scaling and Encoding)

---

### Methodology & Execution

#### 1. Handling Missing Values (Imputation)
| Column | Missing Value Strategy | Rationale |
| :--- | :--- | :--- |
| **Age** | Imputed using the **Median** | Median is robust to outliers, preventing skewing of the age distribution. |
| **Embarked** | Imputed using the **Mode** | Used the most frequent value (S) to fill the two missing rows. |
| **Cabin** | Column **Dropped** | Over 75% of values were missing. Dropping the column avoids generating unreliable features. |

#### 2. Feature Encoding
* **Sex:** Transformed into a numerical feature (`Sex_Encoded`) using **Label Encoding** (Male=1, Female=0).
* **Embarked:** Transformed into numerical columns (`Emb_Q`, `Emb_S`) using **One-Hot Encoding** to prevent the creation of misleading ordinal relationships.
* **Irrelevant Columns:** `Name` and `Ticket` were dropped.

#### 3. Outlier Management
* Outliers in the **Fare** column were identified via visualization (boxplot) and removed using the **Interquartile Range (IQR) method** ($Q3 + 1.5 \times IQR$).
* **Result:** The dataset was reduced from **891 rows** to **775 rows** after removing extreme outliers.

#### 4. Feature Scaling
* The continuous numerical features (**Age** and **Fare**) were scaled using **Standardization (Z-Score Scaling)**. This ensures that the features contribute equally to the distance calculations used by many subsequent ML models.

---

### Key Results
The final `df_clean` DataFrame is free of null values, contains only numerical features, and is ready for model consumption.

* **Initial Row Count:** 891
* **Final Clean Row Count:** 775
