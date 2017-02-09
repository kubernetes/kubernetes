#!/bin/bash

set -ex

kubectl --kubeconfig=/srv/kubernetes/config uncordon $(hostname)
status-set 'active' 'Kubernetes unit resumed'
