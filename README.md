# Data Science & Machine Learning — Hands-On Labs

A progressive series of three hands-on Jupyter Notebook labs covering **NumPy array manipulation**, **Pandas data analysis**, and **data preprocessing for machine learning**. The labs build from foundational numerical computing through tabular data analysis to a full preprocessing pipeline, with dedicated tracks for image and text data.

---

## Overview

This project consists of three self-contained, student-oriented lab notebooks that walk through real-world data science workflows:

1. **Lab 1 — NumPy**: From array fundamentals to image analysis and basic NLP with bag-of-words and cosine similarity.
2. **Lab 2 — Pandas**: Loading, inspecting, filtering, cleaning, grouping, merging, and visualizing tabular data using the classic *tips* restaurant dataset.
3. **Lab 3 — Preprocessing**: Handling missing data, encoding categories, scaling features, detecting outliers, splitting train/test sets, and applying preprocessing to images and text.

Each notebook blends **demonstration cells** (run and observe) with **student practice cells** (write your own code), **predict-then-run exercises**, and **reflection questions** — making them suitable for both classroom instruction and self-paced study.

---

## Learning Objectives

Across the three labs, students learn to:

- Create, inspect, and manipulate NumPy arrays
- Understand array properties (shape, ndim, size, dtype) and broadcasting rules
- Perform indexing, slicing, filtering, reshaping, joining, and splitting on arrays
- Apply element-wise mathematical operations on arrays
- Represent and analyze images as NumPy arrays (shape, RGB channels, brightness, cropping, flipping, thresholding)
- Build bag-of-words vectors and compute cosine similarity with NumPy
- Load, inspect, and explore a real dataset (tips) with Pandas
- Select, filter, sort, and create new columns in DataFrames
- Handle duplicates, missing values, wrong data types, and column renaming
- Use `groupby()` and `agg()` for categorical summarization
- Merge two related DataFrames
- Visualize data with Matplotlib
- Handle missing data using group-wise imputation strategies
- Encode categorical variables with label encoding and one-hot encoding
- Apply min-max scaling and standardization to numeric features
- Detect outliers using the IQR method and handle them via capping (winsorization)
- Split data into train/test sets using `train_test_split`
- Normalize image pixel values from 0–255 to 0–1
- Resize images to a consistent shape for model input
- Clean text data (lowercasing, punctuation removal, tokenization) and encode tokens as numeric IDs

---

## Labs Included

| # | Notebook | Main Topics |
|---|---|---|
| 1 | `NumPy_HandsOn_Array_to_Image_Analysis_Student_Lab_Final.ipynb` | NumPy fundamentals, array creation & properties, reshaping, broadcasting, boolean indexing, image analysis (RGB, cropping, flipping, thresholding), text analysis (tokenization, bag-of-words, cosine similarity) |
| 2 | `Pandas_HandsOn_Student_Lab WEEK2.ipynb` | Pandas DataFrame basics, tips dataset, data inspection, categorical exploration, selecting/filtering, sorting, creating columns, cleaning (duplicates, missing values, dtypes, renaming), groupby aggregation, merging, visualization, image metadata analysis, word frequency tables |
| 3 | `Preprocessing_HandsOn_Student_Lab.ipynb` | Missing data strategies, categorical encoding (label & one-hot), feature scaling (min-max & standardization), outlier detection (IQR) and capping, train/test splitting, image preprocessing (normalization, resizing), text preprocessing (cleaning, tokenization, word-to-ID mapping) |

---

## Lab 1 — NumPy Hands-On: Array to Image Analysis

**Notebook:** `NumPy_HandsOn_Array_to_Image_Analysis_Student_Lab_Final.ipynb`

### NumPy Fundamentals

The lab opens with a **loops vs. vectorization timing comparison** on 1,000,000 elements, demonstrating that NumPy is roughly **20× faster** than a Python list comprehension due to contiguous memory storage and compiled C operations (vectorization).

### Array Creation

Arrays are created from Python lists with `np.array()`. Student practice covers creating arrays like `[5, 10, 15, 20, 25]`.

### Array Properties

Properties covered: `ndim`, `shape`, `size`, and `dtype`. For a 5-element array: `ndim=1`, `shape=(5,)`, `size=5`, `dtype=int32`.

### Special Arrays

`np.zeros((3,3))`, `np.ones((4,4))`, and `np.eye(3)` (identity matrix) are demonstrated, with student practice creating a 5×5 zeros and 2×6 ones array.

### Sequence Generation

- `np.arange(0, 50, 5)` — step-based generation
- `np.linspace(0, 100, 10)` — count-based generation (10 evenly spaced values)
- Student practice: `np.arange(2, 21, 2)` produces `[2, 4, 6, ..., 20]` and `np.linspace(0, 1, 5)` produces `[0, 0.25, 0.5, 0.75, 1]`

### Random Numbers

`np.random.rand(5)` for uniform floats and `np.random.randint(1, 100, 10)` for random integers.

### Reshaping Arrays

`np.arange(12).reshape(3, 4)` demonstrated with valid reshape shapes (2×6, 4×3, 6×2). Invalid reshape `a.reshape(5, 3)` raises `ValueError` because 5×3=15 ≠ 12, confirming that the total number of elements must remain constant.

### Indexing and Slicing

On a 3×3 matrix (`np.arange(1,10).reshape(3,3)`):
- First row: `matrix[0]` → `[1, 2, 3]`
- First column: `matrix[:, 0]` → `[1, 4, 7]`
- Last row: `matrix[2]` → `[7, 8, 9]`
- Middle element: `matrix[1, 1]` → `5`
- Bottom-right 2×2 sub-matrix: `matrix[1:, 1:]`

### Mathematical Operations

Element-wise operations: `marks + 5`, `marks * 2`, and chained operations (add bonus then scale by 1.5).

### Broadcasting

Broadcasting rules explained: dimensions are compatible if equal or one is 1. Examples include scalar across an array (`salary * 1.10`), 1D array across 2D rows (`grid + row_bonus`), and column + row addition producing a 3×3 matrix.

### Filtering Data (Boolean Indexing)

Boolean conditions filter arrays: `marks[marks > 60]` → `[67, 89, 90]`. Student practice includes extracting failing marks (< 40), counting students above 80 with `.sum()` on boolean array, and replacing values with `np.where`.

### Joining and Splitting

- `np.concatenate([a, b])` joins two arrays
- `np.split(x, 3)` splits 12 elements into 3 equal parts
- `np.split(x, 5)` fails with `ValueError` because 12 is not divisible by 5

### Image Analysis Using NumPy

An image (`image.png`, shape `(358, 358, 3)`) is loaded with PIL and converted to a NumPy array. Techniques demonstrated:
- **Image display** with `plt.imshow()` and `plt.axis("off")`
- **Center cropping** to a 200×200 region using calculated offsets
- **Vertical and horizontal flipping** using array slicing (`[::-1]`)
- **RGB channel extraction** — separate R, G, B channels displayed as grayscale heatmaps
- **Brightness analysis** — `np.mean()` computed per channel (overall: ~135.37, Red: ~158.95, Green: ~107.67, Blue: ~139.48)
- **Thresholding** — grayscale conversion via `np.mean(img_array, axis=2)` followed by `np.where(gray > 128, 255, 0)` for binary output; different thresholds (100, 180) tested

### Mini Project: Image Analyzer

Students combine all image techniques into a single program that prints image dimensions, brightness, cropped center, flipped versions, RGB channels, and thresholded output.

### Text Analysis Using NumPy (NLP Track)

- **Tokenization**: `"NumPy makes numerical computing fast and NumPy makes arrays powerful".lower().split()` → list of tokens
- **Vocabulary building**: `np.unique()` on token array (vocabulary size: 8)
- **Bag-of-words vector**: Custom function counts each vocabulary word's occurrences → `[1, 1, 1, 1, 2, 1, 2, 1]` ("makes" and "numpy" each appear twice)
- **Cosine similarity**: Computed via `np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))`. Similar sentences yield ~0.68, unrelated sentences ~0.40.
- **Most frequent words**: Using `np.argsort()[::-1][:3]` to find top-3 words ("python": 3, "is": 3)

---

## Lab 2 — Pandas Hands-On Student Lab

**Notebook:** `Pandas_HandsOn_Student_Lab WEEK2.ipynb`

### Dataset

The **tips** dataset from Seaborn's GitHub repository: 244 rows, 7 original columns (`total_bill`, `tip`, `sex`, `smoker`, `day`, `time`, `size`). Two derived columns are added: `tip_pct` (tip / total_bill) and `tip_per_person` (tip / size).

### Data Loading and Inspection

- Loaded via `pd.read_csv(url)` from the Seaborn repository
- `tips.shape` → `(244, 7)`
- `tips.info()` shows column dtypes and non-null counts
- `tips.describe()` produces summary statistics
- `tips.columns` lists all column names
- `tips.tail()` displays last 5 rows

### Exploring Categorical Data

- `tips['day'].value_counts()` reveals distribution across Thur, Fri, Sat, Sun
- `tips['time'].value_counts()` → Dinner: 176, Lunch: 68
- `tips['smoker'].value_counts(normalize=True) * 100` → No: 61.89%, Yes: 38.11%

### Selecting and Filtering

- Column selection: `tips[['total_bill', 'tip']]`
- Positional indexing: `tips.iloc[0:5]`
- Label-based filtering: `tips.loc[tips['day']=='Sun', ['total_bill', 'tip', 'size']]`
- Boolean filtering: `tips[tips['size'] >= 5]` for large party tables
- Compound conditions: `tips[(tips['smoker']=='Yes') & (tips['time']=='Lunch')]`

### Sorting

Multi-column sorting: `tips.sort_values(['total_bill', 'tip'], ascending=[False, True])`

### Creating New Columns

Using `.assign()` to add `tip_pct` and `tip_per_person`, and direct assignment for `bill_per_person` after filtering (e.g., Saturday tables with size ≥ 3).

### Cleaning Data

- **Duplicates**: 1 duplicate row found and dropped → shape (243, 9)
- **Missing values**: Introduced by setting 4 rows of `total_bill` and 1 row of `day` to `NaN`. Approaches: `dropna()` vs. `fillna()` with column mean
- **Data types**: `size` converted from int64 to int32; `day` converted to `category` dtype
- **Column renaming**: `tips.rename(columns={'size': 'party_size', 'tip_pct': 'tip_percentage'})`

### Grouping and Aggregation

- `tips.groupby('day')['total_bill'].mean()` — average bill per day
- `tips.groupby('day').agg(avg_bill=('total_bill', 'mean'), avg_tip=('tip', 'mean'), n_bills=('total_bill', 'size'))` — multiple aggregations at once
- `tips.groupby('time')['tip_pct'].mean()` — average tip percentage by Lunch vs Dinner

### Merging Tables

Left-join with a `day_benchmark` DataFrame on the `day` column, and a custom `time_rating` DataFrame on the `time` column.

### Visualization

Bar chart of average `total_bill` by `day` using `plt.bar()` with labeled axes and title. Student practice creates a bar chart of average `tip_pct` by `time`.

### Saving Work

`summary.to_csv('day_summary.csv')` and `time_summary.to_csv('my_time_summary.csv')`

### Mini Project A — Image Metadata Analysis

Building a Pandas DataFrame from multiple uploaded images with columns: `filename`, `height`, `width`, `channels`, `avg_brightness`, `avg_red`, `avg_green`, `avg_blue`. Analysis includes finding the brightest image and plotting brightness across all images.

### Mini Project B — Word Frequency Table

Building a `word_freq` DataFrame with `word` and `count` columns, sorting by frequency. Demonstrated with a sentence containing repeated words ("makes", "powerful"). Student practice: top-5 most frequent words, count of hapax legomena (words appearing only once = 19 for the student's paragraph). Cross-sentence comparison table with `difference` column.

---

## Lab 3 — Data Preprocessing Hands-On Student Lab

**Notebook:** `Preprocessing_HandsOn_Student_Lab.ipynb`

### Dataset

The same **tips** dataset (244 rows), with `tip_pct` added as a derived column.

### Missing Data

**Creating missing values:** 15 random rows of `total_bill` set to `NaN` (6.1% missing).

**Strategies covered:**
- **Drop rows** — safe when missing percentage is small and random
- **Fill with mean/median** — appropriate for numeric columns; median is safer when outliers exist
- **Fill with mode** — for categorical columns
- **Fill with group-wise mean** — smarter: `tips_missing.groupby('day')['total_bill'].transform('mean')` preserves day-to-day differences

**Student practice:** 5 missing `tip` values filled with the **median tip grouped by time** (Lunch/Dinner), demonstrating why group-wise imputation preserves group-level variation that a global statistic would flatten.

### Categorical Encoding

**Label encoding (binary categories):**
- `sex`: `{'Male': 0, 'Female': 1}`
- `smoker`: `{'No': 0, 'Yes': 1}`
- `time`: `{'Lunch': 0, 'Dinner': 1}`

**One-hot encoding (nominal categories):**
- `day` encoded via `pd.get_dummies(tips_enc['day'], prefix='day')`, producing columns: `day_Thur`, `day_Fri`, `day_Sat`, `day_Sun`

**Key insight:** Label-encoding `day` as Thur=0, Fri=1, Sat=2, Sun=3 would misleadingly imply numeric ordering — the model might interpret Sun (3) as "greater than" Sat (2). One-hot encoding avoids this for nominal categories.

### Feature Scaling

**Min-Max Scaling** (rescales to 0–1 range):
```python
(series - series.min()) / (series.max() - series.min())
```
Applied to `total_bill` and `size`. Sensitive to outliers because the range is determined by min and max.

**Standardization** (Z-score scaling, mean=0, std=1):
```python
(series - series.mean()) / series.std()
```
Applied to `total_bill`. Verified: mean ≈ 0 (−6.03×10⁻¹⁷), std = 1.0. More robust to outliers than min-max scaling.

**When to prefer each:**
- Min-max scaling when you need a bounded range and there are no extreme outliers
- Standardization when outliers are present or when the algorithm is distance-based (e.g., KNN)

### Outlier Detection and Handling

**IQR method for `total_bill`:**
- Q1 = 25th percentile, Q3 = 75th percentile, IQR = Q3 − Q1
- Bounds: [−2.82, 40.30]
- **9 outliers** detected (bills above 40.30)

**IQR method for `tip`:**
- Bounds: [−0.34, 5.91]
- **9 outliers** detected

**Boxplot visualization** confirms outlier distribution.

**Capping (winsorization):**
```python
tips_capped['tip'] = tips_capped['tip'].clip(lower=lower_bound, upper=upper_bound)
```
After capping: min = 1.000, max = 5.906, mean = 2.950, std = 1.226. Keeps all rows but limits extreme values.

**When capping is preferable:** Preserves data volume; dropping is better when values are clearly errors or invalid observations.

### Train/Test Split

**Features and target:**
- `X` = `['total_bill_scaled', 'size', 'sex_encoded', 'smoker_encoded']`
- `y` = `tip_pct`

**Split with `test_size=0.2`:**
- Train: (195, 4), Test: (49, 4)

**Split with `test_size=0.3`:**
- Train: (170, 4), Test: (74, 4)

**`random_state=42`** ensures reproducibility — removing it produces a different random split each run.

**Why split first:** The test set must remain unseen during training to provide an unbiased evaluation of model performance on new data.

### Mini Project A — Image Preprocessing

- **Image loading** with PIL and conversion to NumPy array
- **Pixel inspection:** Original range 0–255, dtype `uint8`
- **Normalization:** `img_array / 255.0` rescales to 0–1, dtype becomes `float64`
- **Resizing:** `img_pil.resize((128, 128))` — standardizes all images to a consistent shape
- **Shape verification:** Confirmed `(128, 128, 3)` after resize (RGB preserved)
- **Batch processing:** Loop over all uploaded images, normalize and resize each, store results in a list and statistics in a DataFrame

### Mini Project B — Text Preprocessing

- **Lowercasing:** `"NumPy, Pandas, and Scikit-Learn are AMAZING tools!!!".lower()`
- **Punctuation removal:** `re.sub(r'[^a-z0-9\s]', '', cleaned)`
- **Tokenization:** `.split()` → `['numpy', 'pandas', 'and', 'scikitlearn', 'are', 'amazing', 'tools', 'i', 'use', 'numpy', 'every', 'day']`
- **Vocabulary construction:** `sorted(set(tokens))`
- **Word-to-ID mapping:** `{word: idx for idx, word in enumerate(vocabulary)}` → `{'2': 0, '2026': 1, 'amazing': 2, 'built': 3, ...}`
- **Encoding:** `[word_to_id[word] for word in tokens]` → `[8, 7, 6, 2, 4, 3, 0, 9, 5, 1]`
- **Pandas DataFrame representation:** `pd.DataFrame(list(word_to_id.items()), columns=['word', 'id'])`

**Key limitation:** Arbitrary word IDs create misleading numerical relationships — a model might incorrectly interpret `machine=8` as "greater than" `learning=7`, when the IDs are just arbitrary labels.

---

## End-to-End Preprocessing Workflow

### Tabular Data (from Lab 3)

```
Raw Data
  → Inspect (shape, info, describe)
  → Clean Missing Values (drop / mean / group-wise imputation)
  → Encode Categories (label encoding for binary, one-hot for nominal)
  → Handle Outliers (IQR detection → capping/winsorization)
  → Scale Numeric Features (min-max or standardization)
  → Split into Train/Test
  → Model-Ready Data
```

**⚠️ Data Leakage Warning:** In a production pipeline, preprocessing transformations such as scaling and imputation should be **fitted on the training data only** and then **applied** to the test set — rather than fitting on the full dataset before the split. The lab demonstrates the concepts sequentially for clarity, but the proper order is: split first, then fit preprocessing on training data, then transform both sets.

### Image Preprocessing (Mini Project A)

```
Raw Image Files
  → Load with PIL
  → Convert to NumPy Array
  → Normalize Pixels (0–255 → 0–1)
  → Resize to Consistent Shape (128×128)
  → Verify Shape and Pixel Range
  → Model-Ready Image Arrays
```

### Text Preprocessing (Mini Project B)

```
Raw Text
  → Lowercase
  → Remove Punctuation
  → Tokenize (split into words)
  → Build Vocabulary (unique words)
  → Map Words to Numeric IDs
  → Model-Ready Token ID Sequences
```

---

## Key Concepts Learned

| Concept | What I Learned |
|---|---|
| **NumPy Arrays** | NumPy arrays are ~20× faster than Python lists for element-wise operations because they use contiguous memory and compiled C code (vectorization). |
| **Array Properties** | `ndim` gives dimensions, `shape` gives size per dimension, `size` is total elements, `dtype` is the data type. The total element count must stay constant when reshaping. |
| **Broadcasting** | NumPy can operate on arrays of different shapes by "stretching" the smaller array along compatible dimensions (equal or size-1). |
| **Pandas** | DataFrames provide labeled 2D tables with built-in methods for inspection (`info`, `describe`), filtering, grouping, merging, and visualization. |
| **Missing Data** | Group-wise imputation (filling with each group's own mean/median) preserves group differences that a global mean would flatten. |
| **Label Encoding** | Appropriate for binary or ordinal categories (e.g., Male=0, Female=1). |
| **One-Hot Encoding** | Necessary for nominal categories with no natural order (e.g., day of week) to avoid implying false numeric ordering. |
| **Min-Max Scaling** | Rescales to a fixed 0–1 range; sensitive to outliers because the min and max define the range. |
| **Standardization** | Rescales to mean ≈ 0, std ≈ 1; more robust to outliers than min-max scaling. |
| **Outliers (IQR)** | Values beyond 1.5 × IQR from Q1/Q3 are outliers. Capping (winsorizing) keeps all rows by limiting extreme values; dropping is better for clearly invalid data. |
| **Train/Test Split** | The test set must remain unseen during training to provide unbiased evaluation. `random_state` ensures reproducibility. |
| **Image Preprocessing** | Images are NumPy arrays with shape (height, width, channels). Normalization (÷255) and consistent resizing are standard before feeding into models. |
| **Text Preprocessing** | Cleaning (lowercasing, punctuation removal), tokenization, vocabulary building, and word-to-ID encoding are foundational NLP steps. Arbitrary word IDs can mislead models by implying false numeric relationships. |
| **Cosine Similarity** | Measures the angle between two vectors (not their length), making it effective for comparing text of different lengths. Close to 1 = similar, close to 0 = unrelated. |
| **Bag-of-Words** | Represents text as word-count vectors — ignores word order but captures word frequency. |

---

## Technologies & Libraries

| Library | Usage in This Project |
|---|---|
| **Python 3** | Primary programming language |
| **NumPy** | Array creation, manipulation, math operations, broadcasting, image analysis, text vectorization, cosine similarity |
| **Pandas** | DataFrame operations, data loading, inspection, filtering, grouping, merging, cleaning, visualization |
| **Matplotlib** | Plotting bar charts, displaying images, subplots, boxplots |
| **Pillow (PIL)** | Image loading (`Image.open`), resizing (`img.resize`), display |
| **scikit-learn** | `train_test_split` for train/test splitting |
| **re** (Python standard library) | Regular expressions for punctuation removal in text preprocessing |

---

## Results / Verification

### Lab 1 — NumPy
- Vectorization speedup: ~20× faster than Python loops on 1,000,000 elements
- Array `ndim=1, shape=(5,), size=5, dtype=int32`
- `np.arange(12).reshape(3,4)` succeeds; `reshape(5,3)` raises ValueError (15 ≠ 12)
- Image shape: `(358, 358, 3)` — overall brightness: 135.37, Red: 158.95, Green: 107.67, Blue: 139.48
- Cosine similarity between similar sentences: ~0.68; unrelated: ~0.40
- Top-3 words from student's paragraph: "python": 3, "is": 3

### Lab 2 — Pandas
- Dataset shape: `(244, 7)` (original), `(244, 9)` (after adding derived columns)
- Duplicate rows: 1 → shape after dropping: `(243, 9)`
- Missing values introduced: `total_bill` (4 NaN), `day` (1 NaN)
- Dinner: 176 bills (72.1%), Lunch: 68 bills (27.9%)
- Smokers: 38.11%, Non-smokers: 61.89%

### Lab 3 — Preprocessing
- Missing values: 15 in `total_bill` (6.1%), 5 in `tip` — all imputed successfully
- Standardized `total_bill` mean: ≈ 0, std: 1.0
- `total_bill` outlier bounds: [−2.82, 40.30], 9 outliers detected
- `tip` outlier bounds: [−0.34, 5.91], 9 outliers detected
- After capping: tip max reduced to 5.906 (was higher)
- Train/test split (20%): train (195, 4), test (49, 4)
- Train/test split (30%): train (170, 4), test (74, 4)
- Image normalization: pixel range 0–255 → 0.0–1.0, dtype uint8 → float64
- Resized image shape: `(128, 128, 3)`
- Word-to-ID example: `{'2': 0, '2026': 1, 'amazing': 2, 'built': 3, ...}`

---

## Reflections & Key Takeaways

### Why group-wise imputation preserves group differences
Filling missing values with each group's own mean/median (e.g., average tip per time slot) keeps Lunch and Dinner distributions distinct. A global mean would blur these differences and produce less realistic replacements.

### Why nominal categories should not always be label encoded
Label-encoding `day` as Thur=0, Fri=1, Sat=2, Sun=3 implies a numeric order where none exists. A model might learn that Sun > Sat, which is meaningless. One-hot encoding creates separate binary columns without imposing false ordering.

### Difference between min-max scaling and standardization
Min-max scaling maps values to a fixed 0–1 range using the actual min and max, making it sensitive to outliers. Standardization centers values around 0 with unit variance using mean and std, which is more robust when outliers are present.

### Why outlier capping may be preferable to dropping rows
Capping preserves all data points while limiting the influence of extreme values. Dropping rows removes potentially useful information from non-outlier columns. However, dropping is preferable when values are clearly errors or invalid observations.

### Why train/test separation is important
The test set simulates unseen data — training on all data would produce overly optimistic evaluation metrics that don't reflect real-world performance.

### Why preprocessing is part of data work
Preprocessing is not separate from analysis — it shapes what models can learn. Missing data, unscaled features, unencoded categories, and outliers all directly affect model quality.

### Similarities between image and text preprocessing
Both require conversion from raw form to numeric arrays: images are normalized (÷255) and resized; text is cleaned, tokenized, and mapped to numeric IDs. Both aim for consistent, model-ready numerical representations.

### Why arbitrary word IDs can create misleading numerical relationships
Word-to-ID mapping assigns numbers (e.g., machine=8, learning=7) with no inherent meaning. A model might interpret higher IDs as "more important" or assume arithmetic relationships between words, when the IDs are just arbitrary labels.

---

## Project Structure

```text
.
├── NumPy_HandsOn_Array_to_Image_Analysis_Student_Lab_Final.ipynb
├── Pandas_HandsOn_Student_Lab WEEK2.ipynb
├── Preprocessing_HandsOn_Student_Lab.ipynb
├── image.png
├── README.md
└── .gitignore
```
