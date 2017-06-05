#!/bin/sh
set -ux

alias kubectl="kubectl --kubeconfig=/home/ubuntu/config"

kubectl cluster-info > $DEBUG_SCRIPT_DIR/cluster-info
kubectl cluster-info dump > $DEBUG_SCRIPT_DIR/cluster-info-dump
for obj in pods svc ingress secrets pv pvc rc; do
  kubectl describe $obj --all-namespaces > $DEBUG_SCRIPT_DIR/describe-$obj
done
for obj in nodes; do
  kubectl describe $obj > $DEBUG_SCRIPT_DIR/describe-$obj
done
