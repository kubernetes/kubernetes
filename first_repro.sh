#!/bin/bash
#use these commands to reproduce soft lockup.
make clean
make quick-release
./cluster/kube-up.sh
kubectl create -f ./repro_first_trial.yaml
sleep 10
./delete_pods.sh
# once you have the names of the nodes, collect the serial port output to see when soft lockup occurs:
# (on gcp)  gcloud beta compute instances tail-serial-port-output kubernetes-minion-group-[YOUR-XXXX-HERE] --zone=us-central1-b &> node_[YOUR-XXXX-HERE]_serial_out.txt &
# until cat node_[YOUR-XXXX-HERE]_serial_out.txt | grep "soft lockup"; do sleep 10; done
