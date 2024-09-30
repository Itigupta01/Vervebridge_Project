#Vervebridge Internship

#Data Collection

# Loading necessary libraries
library(readr)
library(dplyr) 

# Read the loT sensor data
iot_data <- read_csv("processed_data.csv")

# View the structure of the dataset
str(iot_data)

# Summary statistics
summary(iot_data)

#Data Preprocessing

# Handling missing values (replacing with median)
iot_data <- iot_data %>% 
  mutate_all(~ ifelse(is.na(.), median(., na.rm = TRUE), .))

# Scaling numeric features
library(scales)
iot_data_scaled <- iot_data %>%
  mutate_at(vars(`Temperature (°C)`, `Vibration (mm/s)`, `Pressure (Pa)`, RPM, Temp_Change, Vib_Change), rescale)

# Viewing the scaled data
head(iot_data_scaled)

#Failure Prediction

# Install necessary libraries
library(randomForest)
library(caTools)
set.seed(123)

# Split data into training and testing sets
split <- sample.split(iot_data$`Maintenance Required`, SplitRatio = 0.7)
train_data <- subset(iot_data, split == TRUE)
test_data <- subset(iot_data, split == FALSE)

# Train a Random Forest model
rf_model <- randomForest(`Maintenance Required` ~ `Temperature (°C)` + `Vibration (mm/s)` + `Pressure (Pa)` + RPM + Temp_Change + Vib_Change, 
                         data = train_data, importance = TRUE)

# Predict on the test data
predictions <- predict(rf_model, test_data)


# Evaluate the model
library(caret)
confusionMatrix(predictions, test_data$`Maintenance Required`)

#Anomaly Detection

# Install necessary libraries
library(isotree)

# Scale the data for anomaly detection
iot_data_scaled <- scale(iot_data[,c('Temperature (°C)', 'Vibration (mm/s)', 'Pressure (Pa)', 'RPM')])

# Train the isolation forest model
iso_forest <- isolation.forest(iot_data_scaled)

# Get anomaly scores
anomaly_scores <- predict(iso_forest, iot_data_scaled)

# Add anomaly scores to the data
iot_data$anomaly_score <- anomaly_scores

# Detect anomalies based on a threshold (e.g., score > 0.7)
anomalies <- iot_data %>% filter(anomaly_score > 0.7)

#Real-Time Monitoring

# Function to monitor equipment in real-time
monitor_equipment <- function(new_data) {
  new_data_scaled <- scale(new_data)
  anomaly_score <- predict(iso_forest, new_data_scaled)
  
  if (anomaly_score > 0.7) {
    print("Alert: Anomaly detected! Maintenance required.")
  } else {
    print("Equipment running normally.")
  }
}

# Simulating real-time monitoring with new data
new_data <- data.frame(Temperature = 85, Vibration = 0.5, Pressure = 2, RPM = 1500)
monitor_equipment(new_data)

#Cost-Benefit Analysis

# Simulated cost-benefit analysis
traditional_cost <- 100000  # Cost with traditional maintenance
predictive_cost <- 60000    # Cost with predictive maintenance

savings <- traditional_cost - predictive_cost
savings_percentage <- (savings / traditional_cost) * 100

cat("Cost savings:", savings, "\n")
cat("Savings percentage:", savings_percentage, "%\n")

