#!/bin/bash

# Variables
NODE_NAME="worker"          # Replace with the name of your target worker node
METRICS_SERVER_NAME="metrics-server" # Name of the metrics-server pod or deployment
INTERVAL=1                   # Time interval in seconds between each data collection
DURATION=600                  # Duration in seconds to collect data
TOTAL_NODE_CPU=0             # Sum of all worker node CPU usage data
TOTAL_METRICS_CPU=0          # Sum of all metrics-server CPU usage data
NODE_COUNT=0                 # Number of worker node data points collected
METRICS_COUNT=0              # Number of metrics-server data points collected
OUTPUT_FILE=$(date +%s)  # Output file to store results

# Initialize the output file
echo "Collecting CPU usage data for node '$NODE_NAME' and metrics-server every $INTERVAL second(s) for $DURATION seconds..." > $OUTPUT_FILE

for ((i=0; i < DURATION; i+=INTERVAL)); do
    # Retrieve worker node CPU usage
    NODE_CPU=$(kubectl top node | grep "$NODE_NAME" | awk '{print $2}' | sed 's/m//')
    # Retrieve metrics-server CPU usage via the pod
    METRICS_CPU=$(kubectl top pod -n kube-system | grep "$METRICS_SERVER_NAME" | awk '{print $2}' | sed 's/m//')

    # Check if the values are valid numbers
    if [[ "$NODE_CPU" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        TOTAL_NODE_CPU=$(echo "$TOTAL_NODE_CPU + $NODE_CPU" | bc)
        NODE_COUNT=$((NODE_COUNT + 1))
    fi

    if [[ "$METRICS_CPU" =~ ^[0-9]+(\.[0-9]+)?$ ]]; then
        TOTAL_METRICS_CPU=$(echo "$TOTAL_METRICS_CPU + $METRICS_CPU" | bc)
        METRICS_COUNT=$((METRICS_COUNT + 1))
    fi

    # Log current data points to the output file
    echo "[$(date +'%Y-%m-%d %H:%M:%S')] Node '$NODE_NAME': $NODE_CPU% CPU, Metrics-server: $METRICS_CPU% CPU" >> $OUTPUT_FILE

    # Wait for the specified interval before next data collection
    sleep $INTERVAL
done

# Calculate the average CPU usage
if [ $NODE_COUNT -eq 0 ]; then
    AVG_NODE_CPU="N/A (No valid data points collected)"
else
    AVG_NODE_CPU=$(echo "scale=2; $TOTAL_NODE_CPU / $NODE_COUNT" | bc)
fi

if [ $METRICS_COUNT -eq 0 ]; then
    AVG_METRICS_CPU="N/A (No valid data points collected)"
else
    AVG_METRICS_CPU=$(echo "scale=2; $TOTAL_METRICS_CPU / $METRICS_COUNT" | bc)
fi

# Write final results to the output file
{
    echo ""
    echo "Final Averages:"
    echo "Node '$NODE_NAME' Average CPU Usage: $AVG_NODE_CPU"
    echo "Metrics-server Average CPU Usage: $AVG_METRICS_CPU"
} >> $OUTPUT_FILE

# Notify user of the output
echo "CPU usage report saved to '$OUTPUT_FILE'."
