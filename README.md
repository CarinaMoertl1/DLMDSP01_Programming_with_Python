# Ideal Function Analysis

This project loads the provided CSV datasets, compares the training functions with the available ideal functions, and selects the best matching function for each training function using the least-squares error.

Test data is then compared against the selected functions. A test point is assigned to a function when its deviation is within the calculated acceptance limit.

## How to run the application

### Prerequisites

- Python 3.10 or later
- `pip`, which is included with standard Python installations

### 1. Create and activate a virtual environment

Run these commands from the project root.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install the dependencies

```bash
python -m pip install -r requirements.txt
```

If package installation fails because of a local certificate configuration, add `--trusted-host pypi.org --trusted-host files.pythonhosted.org` to the installation command:

```bash
python -m pip install -r requirements.txt --trusted-host pypi.org --trusted-host files.pythonhosted.org
```

### 3. Check the input data

The default input folder is `datasets/`. It must contain these files:

```text
datasets/
├── train.csv
├── ideal.csv
└── test.csv
```

### 4. Run the analysis

```bash
python main.py
```

The terminal prints the selected ideal functions and the number of mapped test points. The application then writes its generated files to `output/`.

#### Optional: use different input or output folders

```bash
python main.py --data-dir path/to/data --output-dir path/to/output
```

### Run the tests

```bash
python -m unittest discover -s tests -v
```

## Output

Running the application creates an `output/` directory containing the database and visualization.

### SQLite database

`ideal_functions.sqlite` contains the imported training, ideal, and test data, as well as the selected functions and the resulting test-point assignments.

The `test_mappings` table uses the following structure:

| X            | Y            | Delta_Y              | Ideal_Function_No                 |
| ------------ | ------------ | --------------------- | ---------------------------------- |
| test x-value | test y-value | absolute y-deviation | selected ideal function or `NULL` |

Every test observation is kept in the table. Points that do not match any of the selected functions have `NULL` values for the function and deviation.

### Visualization

`visualization.html` provides an interactive Bokeh visualization. It contains an overview of the selected functions and individual views showing the training data, ideal function, assigned test points, and the corresponding deviation limit.

## Data model

`CsvDataSet` is used as the base class for the different dataset types. The training, ideal, and test datasets extend this class with their respective data handling.

The application validates the input structure and reports invalid data or inconsistent processing states using application-specific exceptions.

## Current results

Using the supplied data, the selected functions are:

| Training function | Ideal function |       SSE | Maximum training deviation |
| ------------------ | --------------- | --------: | ---------------------------: |
| y1                 | y13             | 34.080708 |                    0.499221 |
| y2                 | y24             | 33.451761 |                    0.499000 |
| y3                 | y36             | 35.572700 |                    0.498943 |
| y4                 | y40             | 34.998875 |                    0.499779 |

There are 100 test observations in total. 34 are assigned to one of the selected functions, while 66 remain unassigned.

## Development workflow

For changes made in a feature branch:

```powershell
git checkout main
git pull origin main
git checkout -b feature/<new-function>

# make changes and run the tests
git add <changed-files>
git commit -m "Add <short description>"
git push -u origin feature/<new-function>
```

Changes can then be reviewed and merged into `main` through a pull request.