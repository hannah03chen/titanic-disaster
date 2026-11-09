# Titanic Disaster: Survival Prediction with Docker (Python and R)

This repository predicts which passengers survived the sinking of the Titanic (April 15, 1912) using logistic regression. The same analysis is built twice, once in **Python** and once in **R**, and each version runs in its own **Docker container**. You do not need Python or R installed on your computer. You only need Git and Docker.

The data comes from the Kaggle competition [Titanic - Machine Learning from Disaster](https://www.kaggle.com/competitions/titanic). The dataset is **not** included in this repo, so you will download it in Step 2.

---

## Quick start (about 4 steps)

1. Clone the repo and open a terminal in it.
2. Download `train.csv` and `test.csv` from Kaggle into `src/data/`.
3. Run the Python container.
4. Run the R container.

The full instructions for each step are below.

---

## Repository structure

```
titanic-disaster/
├── Dockerfile              # Python container (builds from the repo root)
├── requirements.txt        # Python packages with pinned versions
├── README.md
├── .gitignore
└── src/
    ├── app/
    │   └── main.py         # Python script (data exploration, model, predictions)
    ├── r_app/
    │   ├── Dockerfile      # R container (builds from src/r_app)
    │   ├── install_packages.R
    │   └── titanic_model.R # R script (data exploration, model, predictions)
    └── data/               # YOU add train.csv and test.csv here (ignored by git)
```

---

## Prerequisites

You need two free programs:

| Program | Why | Download |
|---|---|---|
| **Git** | to clone this repo | https://git-scm.com/downloads |
| **Docker Desktop** | to build and run the containers | https://www.docker.com/products/docker-desktop |

You also need a free [Kaggle account](https://www.kaggle.com/) to download the data.

**Before you start, open Docker Desktop and wait until it says Docker is running.** Check it from a terminal with:

```
docker --version
```

If it prints a version number, Docker is installed. If later commands say `Cannot connect to the Docker daemon`, Docker Desktop is not running yet.

---

## Step 1: Clone the repo

Open a terminal (Terminal on Mac, PowerShell on Windows) and run:

```
git clone https://github.com/hannah03chen/titanic-disaster.git
cd titanic-disaster
```

Everything below is run from this folder (the "repo root"). Check that you are in the right place with `ls` (Mac/Linux) or `dir` (Windows). You should see `Dockerfile`, `requirements.txt`, `README.md`, and `src`.

---

## Step 2: Download the data

The dataset is not stored in this repo, so you need to download it yourself.

1. Go to the Kaggle data page: https://www.kaggle.com/competitions/titanic/data
2. Log in to Kaggle. If you see a **Join Competition** or **I Understand and Accept** button, click it and accept the rules. Kaggle will not let you download the files until you do.
3. Click **Download All**. This saves `titanic.zip` (usually in your Downloads folder).
4. Unzip it. You will see three files: `train.csv`, `test.csv`, and `gender_submission.csv`.
5. Copy **`train.csv`** and **`test.csv`** into the `src/data/` folder of this repo. (`gender_submission.csv` is not used.)

**Mac/Linux shortcut**, from the repo root (adjust the path if your zip is somewhere else):

```
unzip ~/Downloads/titanic.zip -d src/data
```

**Check that the files are in the right place** (the output should include `train.csv` and `test.csv`):

```
ls src/data
```

Windows PowerShell: `dir src\data`

> The `.gitignore` excludes `src/data/*.csv`, so these files will never be committed.

---

## Step 3: Run the Python container

All commands are run from the repo root.

**3a. Build the image.** This installs the packages listed in `requirements.txt` (pandas, numpy, scikit-learn, and their dependencies). It takes a minute or two the first time.

```
docker build -t titanic-python .
```

Do not forget the `.` at the end.

**3b. Run the container.** This starts the container, mounts your local `src/data` folder into it, and runs `src/app/main.py`:

Mac / Linux:

```
docker run --rm -v "$(pwd)/src/data:/app/src/data" titanic-python
```

Windows PowerShell:

```
docker run --rm -v "${PWD}/src/data:/app/src/data" titanic-python
```

The data folder is mounted at run time because the dataset is not part of the image.

### What you should see (Python)

The script prints every step it takes to the terminal. The output follows this order:

1. `Container is running!`
2. **Load:** the shape of `train.csv` (891 rows, 12 columns)
3. **Explore:** the first 5 rows, missing values (Age: 177, Cabin: 687, Embarked: 2), and survival rates overall (0.384), by sex (female 0.742, male 0.189), and by passenger class
4. **Adjust:** missing `Age` and `Fare` are filled with the median, `Embarked` with the most common value, `Sex` is encoded as 0/1, `Embarked` is one-hot encoded, and a new `FamilySize` feature is added (SibSp + Parch + 1)
5. **Model:** train.csv is split 80/20 (712 training rows, 179 validation rows), a logistic regression model is trained, and its coefficients are printed
6. **Accuracy and predictions:** training accuracy, the number of passengers in `test.csv` predicted to survive, and the accuracy on the held-out validation rows

Expected results (should match, since the random seed is fixed):

```
Training accuracy: 0.806
Predicted 155 survivors out of 418 test passengers
Held-out validation accuracy: 0.793
```

---

## Step 4: Run the R container

The R version lives in `src/r_app/` and has its own Dockerfile.

**4a. Build the image.** This installs the R packages listed in `install_packages.R` (dplyr and readr). **The first build takes several minutes**, so be patient.

```
docker build -t titanic-r src/r_app
```

**4b. Run the container**, mounting the data folder:

Mac / Linux:

```
docker run --rm -v "$(pwd)/src/data:/app/data" titanic-r
```

Windows PowerShell:

```
docker run --rm -v "${PWD}/src/data:/app/data" titanic-r
```

### What you should see (R)

The R script follows the same steps as the Python one and prints each of them: `R container is running!`, loading and exploring `train.csv`, cleaning (median and mode filling, factor conversion, `FamilySize`), the 80/20 split, the model coefficients, and the accuracy figures and predictions.

Expected results:

```
Training accuracy: 0.809
Predicted 152 survivors out of 418 test passengers
Held-out validation accuracy: 0.765
```

The R and Python numbers differ slightly because each language splits the data randomly in a different way and encodes the features slightly differently.

---

## About the data and the model

- **Data:** 891 passengers in `train.csv` (with a `Survived` column of 0/1) and 418 passengers in `test.csv` (without it).
- **Features used:** passenger class, sex, age, family size, fare, and port of embarkation.
- **Model:** logistic regression (`LogisticRegression` in scikit-learn for Python, `glm(..., family = binomial)` in R).
- **How accuracy is measured:** Kaggle's `test.csv` does **not** include the true `Survived` answers, so its accuracy cannot be calculated. Instead, 20% of `train.csv` is held out and never used for training, and the accuracy on those rows is reported as the accuracy on unseen data. The model still loads `test.csv` and predicts survival for all 418 passengers.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `Cannot connect to the Docker daemon` | Open Docker Desktop and wait until it says it is running, then try again. |
| `FileNotFoundError` for `train.csv` or `test.csv`, or R says `cannot open file 'data/train.csv'` | The data is missing or in the wrong folder. Make sure `train.csv` and `test.csv` are directly inside `src/data/`, and that you ran the command from the repo root with the `-v` option. |
| `docker build` says `no configuration file provided` or cannot find the Dockerfile | You are not in the repo root. Run `cd titanic-disaster`. For R, make sure the command ends in `src/r_app`. |
| The mount fails with a permission or "not shared" error | In Docker Desktop, go to Settings > Resources > File Sharing and make sure the repo's folder is allowed. |
| Windows: `$(pwd)` does not work | Use PowerShell with `${PWD}` as shown above. |
| R build is slow | This is normal on the first build. Later builds reuse the cache. |

---

## Optional: run the Python script without Docker

If you prefer, you can run the Python version in a virtual environment (Python 3.12+ recommended):

```
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
python src/app/main.py
```

The data files still need to be in `src/data/`.

---

## Branches

- `main`: final, working version of the project
- `develop`: integration branch that feature branches were merged into
- Feature branches were used for the project setup, the Python container, and the R container, and each was merged into `develop` through a pull request.
