# Day 4: Python Libraries: NumPy, Pandas & Student Performance Analysis

## 🧠 Technical Takeaways: NumPy Fundamentals
During today's session, I transitioned from using standard Python lists to leveraging **NumPy (Numerical Python)**, the core framework for scientific computing in AI and Machine Learning.
* **Memory Architecture:** Learned that Python lists contain pointers to objects scattered in memory, which slows down calculations. NumPy arrays (`ndarray`) use contiguous memory blocks and compiled C code under the hood, allowing for ultra-fast vectorized arithmetic operations.
* **Vectorized Mathematics:** Practiced element-wise calculations (addition, subtraction, multiplication, and division) between arrays without using slow `for` loops.
* **The Rules of Reshaping:** Mastered the `.reshape()` method. I learned that you can dynamically change a matrix layout (e.g., converting a flat 10-element 1D array into a 2x5 2D grid or a 5x2 2D grid) *only* if the total number of items remains exactly identical.
* **Exclusive Slicing Boundaries:** Deep-dived into the `[start:stop]` boundary rules. Because Python slicing stops *before* the final index, cutting a sub-grid out of a 2D matrix requires adding `+1` to the stopping point or leaving the boundary blank (e.g., `matrix[1:, 1:]`) to safely grab data to the edge.

---

## 📊 Technical Takeaways: Pandas Basics
I explored **Pandas**, the premier library for structural data manipulation and exploratory data analysis (EDA).
* **Core Data Layouts:** Differentiated between a **Series** (a 1-dimensional labeled array representing a single column) and a **DataFrame** (a 2-dimensional spreadsheet-style table with aligned rows and columns).
* **Structural Profiling Tools:** Used programmatic commands to audit datasets instantly:
  - `.info()`: Profiles data types, row counts, and structural memory usage (crucial for noticing if numeric columns are accidentally saved as text).
  - `.describe()`: Automatically calculates key statistical summaries (count, mean, standard deviation, minimum, maximum, and percentiles).
  - `.isnull().sum()`: Loops through columns to count missing (`NaN`) entries, exposing where data is incomplete.
* **Boolean Filtering Masks:** Learned how to apply logic conditions horizontally across a DataFrame to isolate subset records that match specific criteria.

---

## 💼 Detailed Activity & Task Breakdown

### 🛠️ Task 1: NumPy Core Computations (`numpy_tasks.py`)
* **What I Did:** Built a production-ready program executing multidimensional numeric arrays. 
* **Key Steps Implemented:**
  1. Created a 10-element 1D vector and a 2x5 2D matrix structure.
  2. Applied element-wise matrix math including simultaneous division and scalar power calculations.
  3. Extracted aggregate benchmarks across different array axes using `.max()`, `.min()`, `.mean()`, and `.sum()`.
  4. Performed advanced 2D structural transformations, flipping a 2x5 matrix into a 5x2 vertical layout using `.reshape(5, 2)`.
  5. Extracted targeted metrics using multi-axis coordinate indexing to safely isolate numbers inside a grid.

### 🛠️ Task 2: Titanic Dataset Exploratory Analysis (`pandas_tasks.py`)
* **What I Did:** Connected an external script directly to the official **Kaggle Titanic Dataset** to audit real-world passenger records.
* **Key Steps Implemented:**
  1. Configured custom structural layout rules via `pd.set_option()` to prevent the terminal from truncating columns with dots (`...`).
  2. Isolated missing data parameters, discovering that the `Cabin` and `Age` categories contained significant null entries.
  3. Executed an array filtration mask to isolate and extract a sub-table containing only underage passengers (`Age < 18`).
  4. Trimming analysis paths by filtering data features to isolate structural survival trends.

---

## 🎓 Mini-Project Deep-Dive: Student Performance Analysis

### 📋 Project Overview & Dataset Layout
Using a multi-column dataset containing 100 student records (`student_record.csv`), I built a automated pipeline to profile academic performance. The original dataset tracked 10 detailed structural attributes:
- **Identifiers:** `Roll No`, `Student Name`
- **Subject Scores:** `English`, `Mathematics`, `Physics`, `Chemistry`, `Biology`, `Computer`
- **Aggregates:** `Total Marks`, `Percentage`

### 💻 Implementation Steps (`student_analysis.py`)
1. **File I/O Pipeline:** Loaded the tabular structure safely via `pd.read_csv()` using custom directory routing paths.
2. **Automated Summary Audit:** Triggered `.info()` to confirm all subject score fields were properly registered as numeric data types (`int64` / `float64`).
3. **Subject-Specific Loop Analysis:** Programmed an optimized `for` loop that automatically scans the DataFrame's columns, verifies if a subject exists, and uses `.mean()` to compute the exact class average for all 6 subjects independently.
4. **Performance Sorting:** Leveraged `.sort_values(by="Percentage", ascending=False)` to reorganize the table and extracted the top 5 highest-achieving rows using `.head(5)`.
5. **Benchmark Filtration Masking:** Calculated the absolute historical mean percentage of the entire class, then used that single variable to dynamically create a subset table containing only the records of students falling below that class benchmark.
6. **Clean File Export:** Leveraged `.to_csv()` combined with the `index=False` configuration parameter to safely write out a polished summary report file named `processed_student_record.csv` without cluttering the file with extra system row numbers.

### 💡 Key Dataset Insights Discovered
* **Subject Strengths & Weaknesses:** The automated subject looping system allowed for instant identification of the highest-performing subjects compared to core subjects where the overall class average dipped, pinpointing exactly where instructional support is needed.
* **Elite Tracking:** Re-sorting the data by `Percentage` isolated the top 5 students cleanly, showing consistent high marks across both science and humanities courses.
* **Risk Group Isolation:** By setting the global class average percentage as a strict baseline filter, the program successfully isolated a specific cohort of below-average students, enabling targeted academic monitoring and intervention.

---

## ⚠️ Challenges Faced & Structural Resolutions

1. **Local Namespace Import Crashes**
   - *Challenge:* Python threw an `AttributeError: module 'numpy' has no attribute 'array'` error when running the code.
   - *Resolution:* Discovered that naming my script files `numpy.py` and `pandas.py` caused Python to import my empty files instead of the actual installed libraries. Resolved this by renaming them to `numpy_tasks.py` and `pandas_tasks.py`.

2. **Console Wrapping & Table Fragmentation**
   - *Challenge:* The wide 12-column layouts of the Titanic and Student datasets overflowed the narrow VS Code terminal boundaries, causing columns to wrap around unevenly and split into messy, unreadable fragments.
   - *Resolution:* Integrated `pd.set_option('display.max_columns', None)` and `pd.set_option('display.width', 1000)` into the script header, overriding terminal constraints and forcing tables to render on clean, single side-by-side rows.

3. **Dangling Extension Code Bleed**
   - *Challenge:* The script ran into multiple Pylance syntax compilation errors at the bottom of the file (`Unexpected indentation`, `Expected expression`).
   - *Resolution:* Tracked the line bugs down to line 31 using the VS Code "Problems" utility tab. Found that accidental copy-paste markdown formatting characters had bled into the code editor. Cleared the lines cleanly after the `.to_csv()` command to fix the script.
