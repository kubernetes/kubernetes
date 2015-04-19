#!/bin/bash

LOG_FILES=$LOGS_DIR/*

for FILE in $LOG_FILES; do
  echo -e "\n\n\n"
  echo "=================================================================="
  echo " $FILE"
  echo "=================================================================="
  cat $FILE
done
