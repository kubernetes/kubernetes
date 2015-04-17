#!/bin/bash

# This script sets up the guestbook example application in Kubernetes.
# The KUBERENTES_MASTER variable must be set to the URL for kubectl to work.
# The first argument is optional and can be used for debugging.

set -o errexit  # (set -e)

DEBUG=false
if [[ "$1" == "-d" ]] || [[ "$1" == "--debug" ]]; then
  DEBUG=true
  set -o xtrace  # (set -x)
fi
cd /opt/kubernetes/
# Step One Turn up the redis master
kubectl create -f examples/guestbook/redis-master.json
if [[ "${DEBUG}" == true ]]; then
  kubectl get pods
fi
# Step Two: Turn up the master service
kubectl create -f examples/guestbook/redis-master-service.json
if [[ "${DEBUG}" == true ]]; then
  kubectl get services
fi
# Step Three: Turn up the replicated slave pods
kubectl create -f examples/guestbook/redis-slave-controller.json
if [[ "${DEBUG}" == true ]]; then
  kubectl get replicationcontrollers
  kubectl get pods
fi
# Step Four: Create the redis slave service
kubectl create -f examples/guestbook/redis-slave-service.json
if [[ "${DEBUG}" == true ]]; then
  kubectl get services
fi
# Step Five: Create the frontend pod
kubectl create -f examples/guestbook/frontend-controller.json
if [[ "${DEBUG}" == true ]]; then
  kubectl get replicationcontrollers
  kubectl get pods
fi

set +x

echo "# Now run the following commands on your juju client"
echo "juju run --service kubernetes 'open-port 8000'"
echo "juju expose kubernetes"
echo "# Go to the kubernetes public address on port 8000 to see the guestbook application"
