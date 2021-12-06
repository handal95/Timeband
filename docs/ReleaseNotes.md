# Timeband Release Notes
## **TIMEBAND 2.0** 
Forecasting on **multivariable** Time Series data
---
### November 11, 2021 / v2.3
- Stable Implementation of 3 main processes(Train / Clean / Preds)

#### New Features
- `TIMEBANDRunner` is splited Data cleansing and predicting process
  (*Temporarily split for feature addition and refactoring)

- Dataset path and output directory path are merged.
- Processed Data during model execution are saved in file format
- Separation of data setting(`init_dataset`) and data loading(`load_dataset`)

#### Removed
- No more cutoffs option for min-max regularization
- zero imputation is no longer provided within TIMEBAND

---
### November 4, 2021 / v2.2.1
#### New Features
- Refactoring related to config settings
- Modify dashboard visualization (grid and plot order)
- Distinguish between BEST model and regularly saved model
- Use median value instead of band's border for anomaly adjustment
- Visualization options can be set during the learning process

#### Bug Fixes
- Fix last data error in windowing

### November 4, 2021 / Hotfix
- hotfix `adjust` logic in Timeband trainer 
   - Synchronize execution logic with runner 

---
### October 28, 2021 / v2.2
#### New Features
- File output with missing value imputation and anomaly correction
- Label output about missing values and anomalies 
- Modify output directory/file path
- Modify log file from daily record to run-by-run record 
- More CLI argument options

#### Bug Fixes
- Bugfix for Memory leak
- Refactor for Dataset architecture

---

### October 19, 2021 / v2.1
#### New Features
- Colored String options
- CLI argument options
- Timeband Dataset lightweight
- Dataset lightweight

#### Bug Fixes
- Bugfix for model reloading option
- Bugfix for Dashboard prediction lines

- Introduction of encoder decoder architecture
- Implement a predictive visualization dashboard

---

### October 15, 2021 / v2.0 HOTFIX
#### Bug Fixes
- Fix initializing dashboard time index
- Fix closing dashboard window
- Fix small validset index error

---

### October 10, 2021 / v2.0
#### New Features
- Multivariate Data Prediction
- Introduction of encoder decoder architecture
- Implement a predictive visualization dashboard

#### Bug Fixes
- Resolving Mode Collapse

#### ETC
- Update `README.md` about installation
- Update `Release.md` </br>
  For subsequent updates, release notes are made for each version.
---

## **TIMEBAND 1.0** 
Forecasting on singlevariable Time Series data

### September, 2021
#### New Features
- Future data prediction and band formation
- Missing value detection and Imputation on single-variable times series data
- Outlier detection and correction on single-variable time series data
