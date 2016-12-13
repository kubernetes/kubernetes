#!/bin/bash
echo "Starting to delete pods every 10 seconds"
while true
do
  echo "_______________________________"
  kubectl get pods
  kubectl get nodes
  kubectl delete pods --all
  sleep 5
done
