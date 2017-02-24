#!/bin/bash
GCI_IMAGE_PROJECT=container-vm-image-staging
GCI_IMAGE_FAMILY=gci-canary-test
GCI_IMAGE=$(gcloud compute images describe-from-family ${GCI_IMAGE_FAMILY} --project=${GCI_IMAGE_PROJECT} --format="value(name)")
DOCKER_VERSION=$(curl -fsSL --retry 3 https://api.github.com/repos/docker/docker/releases | tac | tac | grep -m 1 "\"tag_name\"\:" | grep -Eo "[0-9\.rc-]+")
GCI_CLOUD_INIT=test/e2e_node/jenkins/gci-init.yaml

# Render the test config file
GCE_IMAGE_CONFIG_PATH=`mktemp`
CONFIG_FILE=test/e2e_node/jenkins/docker_validation/perf-config.yaml
cp $CONFIG_FILE $GCE_IMAGE_CONFIG_PATH
sed -i -e "s@{{IMAGE}}@${GCI_IMAGE}@g" $GCE_IMAGE_CONFIG_PATH
sed -i -e "s@{{IMAGE_PROJECT}}@${GCI_IMAGE_PROJECT}@g" $GCE_IMAGE_CONFIG_PATH
sed -i -e "s@{{METADATA}}@user-data<${GCI_CLOUD_INIT},gci-docker-version=${DOCKER_VERSION},gci-update-strategy=update_disabled@g" $GCE_IMAGE_CONFIG_PATH

GCE_HOSTS=
GCE_ZONE=us-central1-f
GCE_PROJECT=k8s-jkns-ci-node-e2e
CLEANUP=true
GINKGO_FLAGS='--skip="\[Flaky\]"'
PARALLELISM=1
