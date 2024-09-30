import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.ensemble import IsolationForest

# Read the Excel file
file_path = r'C:\Users\Iti\Downloads\processed_data.xlsx'  # Ensure you use the correct path
data = pd.read_excel(file_path, engine='openpyxl')

# Strip any whitespace from column names
data.columns = data.columns.str.strip()

# Ensure 'Timestamp' column is in datetime format
if 'Timestamp' in data.columns:
    data['Timestamp'] = pd.to_datetime(data['Timestamp'])
    data.set_index('Timestamp', inplace=True)  # Set 'Timestamp' as the index
else:
    print("No 'Timestamp' column found, please check the dataset.")

# Now data.index is a DatetimeIndex, so we can extract 'Hour' and 'Day'
data['Hour'] = data.index.hour
data['Day'] = data.index.dayofweek

# Step 1: Data Scaling
scaler = StandardScaler()
data[['Temperature (Â°C)', 'Vibration (mm/s)', 'Pressure (Pa)']] = scaler.fit_transform(
    data[['Temperature (Â°C)', 'Vibration (mm/s)', 'Pressure (Pa)']]
)

# Handle Missing Values
data.fillna(method='ffill', inplace=True)

# Step 3: Data Splitting
# Check if there's a column indicating equipment failure
target_column = 'Failure'  # Replace 'Failure' with the actual column name if found

if target_column in data.columns:
    y = data[target_column]
else:
    print(f"Target column '{target_column}' not found.")
    y = None

if y is not None:
    # Supervised learning if the failure column exists
    X = data[['Temperature (Â°C)', 'Vibration (mm/s)', 'Pressure (Pa)', 'Hour', 'Day']]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Random Forest Classifier
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Make predictions
    y_pred = rf_model.predict(X_test)

    # Evaluate the model
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 5: Anomaly Detection (Isolation Forest)
X = data[['Temperature (Â°C)', 'Vibration (mm/s)', 'Pressure (Pa)', 'Hour', 'Day']]

iso_forest = IsolationForest(contamination=0.01, random_state=42)
iso_forest.fit(X)

# Predict anomalies
anomalies = iso_forest.predict(X)

# Label anomalies
data['Anomaly_Status'] = ['Anomaly' if x == -1 else 'Normal' for x in anomalies]

# Display results
print(data[['Temperature (Â°C)', 'Vibration (mm/s)', 'Pressure (Pa)', 'Anomaly_Status']].head())

# Plot anomalies (Temperature vs. Vibration)
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Temperature (Â°C)'], y=data['Vibration (mm/s)'], hue=data['Anomaly_Status'], palette=['red', 'blue'])
plt.title('Anomaly Detection (Red = Anomaly, Blue = Normal)')
plt.xlabel('Temperature (Â°C)')
plt.ylabel('Vibration (mm/s)')
plt.show()
