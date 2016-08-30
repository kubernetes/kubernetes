#!/bin/bash

set -ex

kubectl --kubeconfig=/srv/kubernetes/config cordon $(hostname)
kubectl --kubeconfig=/srv/kubernetes/config drain $(hostname) --force
status-set 'waiting' 'Kubernetes unit paused'
