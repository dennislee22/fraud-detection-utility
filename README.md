# Detecting Electricity Fraud with Unsupervised Learning

![fraud-utility](https://github.com/user-attachments/assets/918877db-e02d-42f0-94c5-12b523f10376)

Electricity fraud, from meter tampering to illegal connections, costs utility companies millions of dollars annually. Catching these fraudsters is a classic cat-and-mouse game, but ML offers a powerful new weapon. This article illustrates an unsupervised anomaly detection system that can flag suspicious electricity consumption patterns.
The [script](fraud-map.ipynb) covers:

1. Simulating a realistic dataset.
2. Engineering features to highlight abnormal behavior.
3. Training an `IsolationForest` model and understanding why it's the right tool for the job.
4. Visualizing the results with Matplotlib and an interactive Folium map.

## Step 1: You Can't Build a Model Without Data
The first hurdle in many ML projects is the lack of a good dataset. So, we created our own. Our simulation script generates a rich dataset of 36 months of electricity usage for hundreds of consumers. The key to a good simulation is realism. We didn't just generate random numbers. Our script accounts for:

 - Community Structure: Consumers are assigned a postcode and building_type (e.g., 'bungalow', 'penthouse').
 - Base Usage: Each building type has a different baseline electricity consumption.
 - Potential Fraudsters: We secretly designate a small percentage of consumers as "fraudulent." After a year of normal behavior, their consumption patterns change drastically to simulate either meter tampering (abnormally low usage) or energy theft (abnormally high usage).

## Step 2: Feature Engineering
A machine learning model rarely learns from raw data. We need to create features that explicitly highlight the behavior we want to detect. For fraud detection, the most powerful feature isn't the raw kWh usage, but how much a consumer deviates from their peers.

The code does the following:

1. Calculates the "Community Norm": For every postcode, building_type, and month, it calculates the average electricity usage. This value represents the expected, normal consumption for a specific peer group at a specific time of year.
2. Calculates Deviation: It then computes a deviation_pct for every single monthly record. This feature answers the question: "How different was this consumer's usage from their community's norm this month?"
3. Creates 12-Month Windows: A single month's deviation isn't enough. We need to see a pattern. The script creates thousands of training samples by taking 12-month sliding windows of the deviation_pct for every consumer.

The final output is a feature matrix where each row represents a consumer's year-long behavior, described by 12 deviation values. This is what our model will learn from.

## Step 3: Model Selection - Why IsolationForest?
For unsupervised anomaly detection, `IsolationForest` is an excellent and highly recommended algorithm.

How it Works:
Imagine trying to describe a single "abnormal" data point in a large crowd. It's often easier to isolate it than to describe the entire "normal" crowd. `IsolationForest` is built on this principle. It creates a forest of random decision trees. The core idea is that anomalies are "few and different," meaning they are easier to separate from the rest of the data. In a random tree, a fraudulent data point will likely be isolated with fewer splits, resulting in a shorter path from the root of the tree. The model calculates an "anomaly score" based on this average path length.

Why it's a Great Choice:
Designed for Anomalies: Unlike clustering algorithms (like K-Means) that find groups, `IsolationForest` is specifically built to find outliers.
Efficient: It's computationally fast and scales well to large datasets.
No Assumption of Normality: It doesn't require the "normal" data to follow a specific statistical distribution.
The model is trained on our matrix of 12-month deviation vectors and saved to a file (fraud-model.pkl) for later use.

## Step 4: Making Predictions and Visualizing Results
With a trained model, we can now analyze new data. Our prediction scripts load the model and run new consumer data through the same feature engineering pipeline to generate a 12-month deviation vector. The model then assigns an anomaly score.
A simple report is good, but visualizations are better.

### The Scatter Plot
Normal Consumers (Blue): Cluster tightly around the 0% deviation line, which represents the community average.
Negative Outliers (Red): These are consumers with significantly lower usage than their peers (e.g., -80% deviation). This is a strong indicator of meter tampering.
Positive Outliers (Orange): These consumers use significantly more energy than their peers, which could indicate energy theft or other undeclared activities.
<img width="1439" alt="image" src="https://github.com/user-attachments/assets/72d3679f-956f-484c-b0ab-040dcc38c1a0" />

The Interactive Map
To give our analysis a real-world context, we used folium to plot every consumer on a map of Kuala Lumpur. This is incredibly powerful for operational teams who need to investigate flagged locations. By using MarkerCluster, we can handle hundreds of points without cluttering the map.

When you zoom in, you can click on any consumer to see their status and anomaly score, providing an intuitive and actionable view of the data.
<img width="1443" alt="image" src="https://github.com/user-attachments/assets/db7dae6e-4bf8-459a-9460-a97ced48baf6" />

As we don't have "true" labels to compare against, the model's performance is judged by its ability to assign low anomaly scores to the outliers we intentionally created while keeping the scores for normal consumers close to zero.

The contamination parameter in the `IsolationForest` model acts as a sensitivity knob. We set it to 0.05 (5%), telling the model to set its anomaly threshold such that it flags about 5% of the data as anomalous. This is why our report might show 10 frauds when we only created 7—the model also found 3 other consumers whose usage was, by pure chance, statistically unusual enough to be flagged.

This project demonstrates a complete workflow for building an effective, unsupervised fraud detection system. By focusing on strong feature engineering and choosing the right algorithm for the job, we were able to build a model that not only identifies suspicious behavior but also provides interpretable and actionable results through powerful visualizations.









