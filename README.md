# Optimizing-Rate-of-Penetration-in-Geothermal-Drilling-A-Digital-Twin-Approach

Let’s explore something interesting together. In this project, we developed a machine learning digital twin using Intel-optimized XGBoost and daal4py to simulate and optimize the Rate of Penetration (ROP) in geothermal drilling. We leveraged SHAP for Explainable AI (XAI) to interpret model predictions, providing insights into the impact of input features. 

I would like to thank [Digital Twin Design for Renewable Energy Exploration](https://medium.com/intel-analytics-software/digital-twin-design-for-renewable-energy-exploration-2fcc4c545f8d) for providing the foundational concepts and methodologies used in this project.

<hr>

## **Digital Twin Design for Renewable Energy Exploration. Building Digital Twins with XGBoost.** 

## Introduction

Digital twins are virtual representations of physical objects or systems. These representations allow us to monitor the digital twin when exposed to external/internal variables and predict how it will perform in a production environment.

### Geothermal Energy Exploration

Geothermal energy exploration involves identifying and utilizing the Earth’s natural heat as a sustainable energy resource. Geoscientists focus on locating geothermal systems that have sufficient heat to justify the investment in drilling wells and constructing power plants. These systems can be categorized into:

- **In-field systems:** Located within known geothermal areas.
- **Near-field systems:** Adjacent to known geothermal areas.
- **Deep geothermal systems:** Found at greater depths, often requiring more advanced drilling techniques.

<img width="695" alt="Screenshot 2024-11-01 at 12 26 43 PM" src="https://github.com/user-attachments/assets/64cc126c-f779-4a25-ace4-a62d00ecd9c9">

### Drilling Challenges

Drilling for geothermal energy, especially in high-pressure and high-temperature environments, presents several challenges:

- **Specialized Tools:** The extreme conditions require tools that can withstand high temperatures and pressures. This includes drill bits, casing, and drilling fluids designed to operate effectively under these conditions.
- **Temperature Requirements:** For flash and dry steam geothermal power plants, temperatures of at least 350°F (about 177°C) are needed. Such temperatures are typically found at depths greater than 10,000 feet (approximately 3,048 meters).
- **Sensor Durability:** High temperatures and pressures can damage sensors, making it difficult to monitor drilling conditions accurately.

<img width="552" alt="Screenshot 2024-11-01 at 12 27 08 PM" src="https://github.com/user-attachments/assets/aa64ae22-acbf-4436-8644-d485a1115427">

### Control Parameters

To optimize drilling conditions and ensure efficient and safe operations, several control parameters are monitored and adjusted:

- **Weight on Bit (WOB):** This is the downward force exerted on the drill bit. Proper WOB ensures effective penetration of the rock without damaging the bit.
- **Rotations Per Minute (RPM):** This measures the number of complete rotations made by the drill bit per minute. Adjusting RPM helps in managing the rate of penetration and the wear on the drill bit.
- **Flow Rate (FLOW):** This is the volume of liquid or slurry that moves through the drilling pipe per unit of time. Proper flow rate is crucial for cooling the drill bit, removing cuttings from the wellbore, and maintaining well pressure.
- **Torque (TOR):** This is the rotational force applied between the drill string and the rock. Managing torque is essential to prevent sticking of the drill string and to ensure smooth drilling operations.

### Building an AI Digital Twin Model

The goal of creating an AI digital twin model is to simulate the performance of geothermal wells. This involves:

- **Identifying Inputs and Outputs:** Understanding the physical system by identifying key inputs (e.g., WOB, RPM, FLOW, TOR) and outputs (e.g., rate of penetration, temperature, pressure).
- **Data Collection:** Gathering data from sensors and historical drilling operations to train the AI model.
- **Modeling and Simulation:** Using AI techniques to create a digital representation of the well, which can predict performance under various conditions and optimize drilling parameters.

<hr>

## Digital Twin Concept

A **digital twin** is a virtual representation of a physical system or process. It uses real-time data and simulations to mirror the behavior and performance of its real-world counterpart. In the context of geothermal energy exploration, a digital twin can be used to model and optimize the performance of geothermal wells.

### How Digital Twins Work

1. **Data Collection**: Sensors and monitoring equipment collect data from the physical system. For geothermal wells, this includes data on WOB, RPM, FLOW, TOR, temperature, pressure, and other relevant parameters.
2. **Modeling**: This data is used to create a digital model of the well. The model incorporates the physical characteristics of the well and the surrounding geological formations.
3. **Simulation**: The digital twin can simulate various scenarios, such as changes in drilling parameters or unexpected conditions. This helps in predicting the performance of the well under different circumstances.
4. **Real-Time Monitoring**: The digital twin continuously updates with real-time data from the physical well. This allows for ongoing monitoring and optimization of drilling operations.
5. **Optimization**: By analyzing the data and simulations, the digital twin can suggest adjustments to drilling parameters to improve efficiency, reduce costs, and enhance safety.

### Benefits of Digital Twins in Geothermal Drilling

- **Predictive Maintenance**: By monitoring the condition of drilling equipment and predicting potential failures, digital twins can help in scheduling maintenance before issues arise.
- **Performance Optimization**: Digital twins can identify the optimal settings for WOB, RPM, FLOW, and TOR to maximize drilling efficiency and minimize wear on equipment.
- **Risk Management**: Simulating different scenarios helps in understanding potential risks and developing strategies to mitigate them.
- **Cost Reduction**: By optimizing drilling operations and reducing downtime, digital twins can significantly lower the overall cost of geothermal energy exploration.

### Application in Geothermal Energy

In geothermal energy exploration, digital twins are particularly valuable because they allow for:

- **Enhanced Decision-Making**: Providing a comprehensive view of the well's performance and potential issues.
- **Improved Safety**: Monitoring high-pressure, high-temperature environments in real-time to prevent accidents.
- **Sustainability**: Ensuring that geothermal wells are drilled and operated in the most efficient and environmentally friendly manner.

By leveraging digital twin technology, geoscientists and engineers can better understand and manage the complexities of geothermal drilling, leading to more successful and cost-effective energy production.

<hr>

## Exploratory Data Analysis

The **Utah FORGE** (Frontier Observatory for Research in Geothermal Energy) project is a significant initiative aimed at advancing geothermal energy technologies. It serves as a dedicated underground field laboratory where various technologies related to enhanced geothermal systems (EGS) are developed, tested, and accelerated.

### The Dataset
The dataset you mentioned is from the drilling of the **Utah FORGE 58–32 MU-ESW1 well** near Roosevelt Hot Springs. This dataset is collected using an **Electronic Drilling Recorder (EDR)**, which is a system that captures and records various drilling parameters in real-time.

### Key Parameters in the Dataset
The dataset is recorded at one-foot intervals and covers depths from **85.18 feet to 7536.25 feet**. Here are some of the key parameters included:

- **Depth**: The vertical distance from the surface to the point of measurement.
- **Rate of Penetration (ROP)**: The speed at which the drill bit advances through the rock, usually measured in feet per hour.
- **Weight on Bit (WOB)**: The amount of downward force applied to the drill bit.
- **Hookload**: The total weight suspended from the hook, including the drill string and any additional equipment.
- **Temperatures In and Out**: The temperature of the drilling fluid entering and exiting the wellbore.
- **Pump Pressure**: The pressure exerted by the pumps circulating the drilling fluid.
- **Flows In and Out**: The volume of drilling fluid being pumped into and out of the well.
- **H2S**: The concentration of hydrogen sulfide gas, which is a hazardous gas that can be encountered during drilling.

### Purpose of the Data
This data is crucial for several reasons:
1. **Monitoring and Control**: Real-time monitoring of these parameters helps in controlling the drilling process, ensuring safety, and optimizing performance.
2. **Training Models**: The data can be used to train machine learning models to predict and optimize drilling performance, detect anomalies, and improve overall efficiency.
3. **Research and Development**: It supports research into new drilling technologies and methods, contributing to the advancement of geothermal energy extraction.
4. **Application in Enhanced Geothermal Systems (EGS)**: Enhanced Geothermal Systems involve creating artificial reservoirs in hot rock formations by injecting fluid to enhance permeability. The data from the EDR helps in understanding the subsurface conditions and optimizing the drilling process to create these reservoirs effectively.

### Overview of Data
![image](https://github.com/user-attachments/assets/b3af0baa-6a1e-4c48-8425-2ab932c1d997)

<hr>

## Analysis and WalkThrough

### Libraries

```python
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
import os
import time
import sys
import joblib
import shap
import numpy as np
import xgboost as xgb
from sklearn import linear_model
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
sns.set(style="darkgrid")

import warnings
warnings.filterwarnings('ignore')
```

### Utilities to handle various aspects of XGBoost and Daal4Py modeling and optimizations.

- **prepare_data**: Splits the dataset into training and testing sets.
- **linreg**: Trains and evaluates a linear regression model, optionally using Intel's optimizations.
- **XGBHyper_train**: Trains an XGBoost model with hyperparameter tuning.
- **XGBReg_train**: Trains an XGBoost model with specified parameters.
- **XGB_predict**: Makes predictions using a trained XGBoost model and calculates the mean squared error.
- **XGB_predict_daal4py**: Makes predictions using a trained XGBoost model with daal4py optimizations.
- **XGB_predict_aug**: Makes predictions using a trained XGBoost model with optional daal4py optimizations.
- **data_proc**: Processes raw data by applying a rolling mean and optional interpolation, and plots the processed data.

### Data Pre-Processing 

For digital twins, we must first define the causal relationships within the modeled system. To achieve this, we will separate the inputs and outputs of the drill rig. This leaves WOB, TOR, RPM, and FLOW as inputs and ROP as our sole output. **ROP defines the amount of downward force applied to rock during drilling.** Lastly, we will remove the first 1000ft of data to focus on areas where drilling is slow and can be improved, like zones of high pressure and temperature.

```python
geothermal_drilling_data_1000ft = geothermal_drilling_data_raw.loc[geothermal_drilling_data_raw['Depth'] > 1000, :]
geothermal_drilling_data_1000ft.head(10)
final_features = ['ROP(1 ft)', 'WOB (k-lbs)', 'Surface Torque (psi)', 'Rotary Speed (rpm)','Flow In (gpm)']
geothermal_data_final = data_proc(geothermal_drilling_data_1000ft, final_features, depth_var='Depth', dropna=True, rolling_win=10, interpolate='cubic')
```
![image](https://github.com/user-attachments/assets/e25450e9-c26b-4032-a94f-54a4458f4d6b)

### Feature Analysis 

**Pearson correlation matrix**

- **Positive Relation:** Flow in, rotary speed, and ROP show moderate positive correlations. This makes sense because fluid flow through the drill string helps spin the bit, leading to higher rotation speeds and rates of penetration. 
- **Negative Relation:** Between ROP and Depth. As depth increases, the rate of penetration (ROP) decreases due to harder rock formations, higher pressure and temperature, and equipment wear.
- **Positive Relation:** Between WOB and Depth. As depth increases, the weight on bit (WOB) increases to counteract the reduced ROP and overcome greater resistance from deeper formations.

**Pairplot Analysis**

Next, we will perform a pair plot analysis, which uses a “small multiple” approach to visualize the univariate distribution of all variables in a dataset along with all of their pairwise relationships. From the diagonal histograms, we see various complex multi-modal and skewed distributions. The kernel density maps associated with depth indicate a complex distribution of clusters for each control parameter, representing multiple rock types and the parameters used to drill through them. 

A pair plot analysis is particularly useful for visualizing these relationships, as it reveals complex, multi-modal, and skewed distributions through diagonal histograms. The kernel density maps for depth highlight clusters for each control parameter, indicating the presence of various rock types and the drilling parameters employed. 

<hr>







