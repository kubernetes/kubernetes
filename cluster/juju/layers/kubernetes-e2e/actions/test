#!/bin/bash

set -ex

# Grab the action parameter values
FOCUS=$(action-get focus)
SKIP=$(action-get skip)
PARALLELISM=$(action-get parallelism)

if [ ! -f /home/ubuntu/.kube/config ]
then
  action-fail "Missing Kubernetes configuration."
  action-set suggestion="Relate to the certificate authority, and kubernetes-master"
  exit 0
fi

# get the host from the config file
SERVER=$(cat /home/ubuntu/.kube/config | grep server | sed 's/    server: //')

ACTION_HOME=/home/ubuntu
ACTION_LOG=$ACTION_HOME/${JUJU_ACTION_UUID}.log
ACTION_LOG_TGZ=$ACTION_LOG.tar.gz
ACTION_JUNIT=$ACTION_HOME/${JUJU_ACTION_UUID}-junit
ACTION_JUNIT_TGZ=$ACTION_JUNIT.tar.gz

# This initializes an e2e build log with the START TIMESTAMP.
echo "JUJU_E2E_START=$(date -u +%s)" | tee $ACTION_LOG
echo "JUJU_E2E_VERSION=$(kubectl version | grep Server | cut -d " " -f 5 | cut -d ":" -f 2 | sed s/\"// | sed s/\",//)" | tee -a $ACTION_LOG
ginkgo -nodes=$PARALLELISM $(which e2e.test) -- \
  -kubeconfig /home/ubuntu/.kube/config \
  -host $SERVER \
  -ginkgo.focus $FOCUS \
  -ginkgo.skip "$SKIP" \
  -report-dir $ACTION_JUNIT 2>&1 | tee -a $ACTION_LOG

# This appends the END TIMESTAMP to the e2e build log
echo "JUJU_E2E_END=$(date -u +%s)" | tee -a $ACTION_LOG

# set cwd to /home/ubuntu and tar the artifacts using a minimal directory
# path. Extracing "home/ubuntu/1412341234/foobar.log is cumbersome in ci
cd $ACTION_HOME/${JUJU_ACTION_UUID}-junit
tar -czf $ACTION_JUNIT_TGZ *
cd ..
tar -czf $ACTION_LOG_TGZ ${JUJU_ACTION_UUID}.log

action-set log="$ACTION_LOG_TGZ"
action-set junit="$ACTION_JUNIT_TGZ"
