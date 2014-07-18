#!/bin/bash

SCRIPT_DIR=$(CDPATH="" cd $(dirname $0); pwd)
source $SCRIPT_DIR/../release/config-azure.sh
source $SCRIPT_DIR/util.sh


echo "Bringing down cluster"
azure vm delete $MASTER_NAME -b -q
for (( i=0; i<${#MINION_NAMES[@]}; i++)); do
    azure vm delete ${MINION_NAMES[$i]} -b -q
done
