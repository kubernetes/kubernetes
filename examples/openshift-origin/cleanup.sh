#!/bin/bash

# Copyright 2014 The Kubernetes Authors All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Cleans up resources from the example, assumed to be run from Kubernetes repo root
echo
echo
export OPENSHIFT_EXAMPLE=$(pwd)/examples/openshift-origin
export OPENSHIFT_CONFIG=${OPENSHIFT_EXAMPLE}/config

echo "===> Removing the OpenShift namespace:"
kubectl delete namespace openshift-origin
echo

echo "===> Removing local files:"
rm -rf ${OPENSHIFT_CONFIG}
rm ${OPENSHIFT_EXAMPLE}/openshift-startup.log
rm ${OPENSHIFT_EXAMPLE}/secret.json
touch ${OPENSHIFT_EXAMPLE}/secret.json
echo

echo "===> Restoring changed YAML specifcations:"
if [ -f "${OPENSHIFT_EXAMPLE}/etcd-controller.yaml.bak" ]; then
	rm ${OPENSHIFT_EXAMPLE}/etcd-controller.yaml
	mv -v ${OPENSHIFT_EXAMPLE}/etcd-controller.yaml.bak ${OPENSHIFT_EXAMPLE}/etcd-controller.yaml
else
	echo "No changed specifications found."
fi
echo

echo Done.
