#!/bin/bash

# Copyright 2014 The Kubernetes Authors.
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

set -e

# Creates resources from the example, assumed to be run from Kubernetes repo root
echo
echo "===> Initializing:"
if [ ! $(which python) ]
then
	echo "Python is a prerequisite for running this script. Please install Python and try running again."
	exit 1
fi

if [ ! $(which gcloud) ]
then
	echo "gcloud is a prerequisite for running this script. Please install gcloud and try running again."
	exit 1
fi

gcloud_instances=$(gcloud compute instances list | grep "\-master")
if [ -z "$gcloud_instances" ] || [ -z "${KUBE_GCE_INSTANCE_PREFIX}" ]
then
	echo "This script is only able to supply the necessary serviceaccount key if you are running on Google"
	echo "Compute Engine using a cluster/kube-up.sh script with KUBE_GCE_INSTANCE_PREFIX set. If this is not"
	echo "the case, be ready to supply a path to the serviceaccount public key."
	if [ -z "${KUBE_GCE_INSTANCE_PREFIX}" ]
	then
		echo "Please provide your KUBE_GCE_INSTANCE_PREFIX now:"
		read KUBE_GCE_INSTANCE_PREFIX
	fi
fi

export OPENSHIFT_EXAMPLE=$(pwd)/examples/openshift-origin
echo Set OPENSHIFT_EXAMPLE=${OPENSHIFT_EXAMPLE}
export OPENSHIFT_CONFIG=${OPENSHIFT_EXAMPLE}/config
echo Set OPENSHIFT_CONFIG=${OPENSHIFT_CONFIG}
mkdir ${OPENSHIFT_CONFIG}
echo Made dir ${OPENSHIFT_CONFIG}
echo

echo "===> Setting up OpenShift-Origin namespace:"
kubectl create -f ${OPENSHIFT_EXAMPLE}/openshift-origin-namespace.yaml
echo

echo "===> Setting up etcd-discovery:"
# A token etcd uses to generate unique cluster ID and member ID. Conforms to [a-z0-9]{40}
export ETCD_INITIAL_CLUSTER_TOKEN=$(python -c "import string; import random; print(''.join(random.SystemRandom().choice(string.ascii_lowercase + string.digits) for _ in range(40)))")

# A unique token used by the discovery service. Conforms to etcd-cluster-[a-z0-9]{5}
export ETCD_DISCOVERY_TOKEN=$(python -c "import string; import random; print(\"etcd-cluster-\" + ''.join(random.SystemRandom().choice(string.ascii_lowercase + string.digits) for _ in range(5)))")
sed -i.bak -e "s/INSERT_ETCD_INITIAL_CLUSTER_TOKEN/\"${ETCD_INITIAL_CLUSTER_TOKEN}\"/g" -e "s/INSERT_ETCD_DISCOVERY_TOKEN/\"${ETCD_DISCOVERY_TOKEN}\"/g" ${OPENSHIFT_EXAMPLE}/etcd-controller.yaml

kubectl create -f ${OPENSHIFT_EXAMPLE}/etcd-discovery-controller.yaml --namespace='openshift-origin'
kubectl create -f ${OPENSHIFT_EXAMPLE}/etcd-discovery-service.yaml --namespace='openshift-origin'
echo

echo "===> Setting up etcd:"
kubectl create -f ${OPENSHIFT_EXAMPLE}/etcd-controller.yaml --namespace='openshift-origin'
kubectl create -f ${OPENSHIFT_EXAMPLE}/etcd-service.yaml --namespace='openshift-origin'
echo

echo "===> Setting up openshift-origin:"
kubectl config view --output=yaml --flatten=true --minify=true > ${OPENSHIFT_CONFIG}/kubeconfig
kubectl create -f ${OPENSHIFT_EXAMPLE}/openshift-service.yaml --namespace='openshift-origin'
echo

export PUBLIC_OPENSHIFT_IP=""
echo "===> Waiting for public IP to be set for the OpenShift Service."
echo "Mistakes in service setup can cause this to loop infinitely if an"
echo "external IP is never set. Ensure that the OpenShift service"
echo "is set to use an external load balancer. This process may take" 
echo "a few minutes. Errors can be found in the log file found at:"
echo ${OPENSHIFT_EXAMPLE}/openshift-startup.log
echo "" > ${OPENSHIFT_EXAMPLE}/openshift-startup.log
while [ ${#PUBLIC_OPENSHIFT_IP} -lt 1 ]; do
	echo -n .
	sleep 1
	{
		export PUBLIC_OPENSHIFT_IP=$(kubectl get services openshift --namespace="openshift-origin" --template="{{ index .status.loadBalancer.ingress 0 \"ip\" }}")
	} >> ${OPENSHIFT_EXAMPLE}/openshift-startup.log 2>&1
	if [[ ! ${PUBLIC_OPENSHIFT_IP} =~ ^([0-9]{1,3}\.){3}[0-9]{1,3}$ ]]; then
		export PUBLIC_OPENSHIFT_IP=""
	fi
done
echo
echo "Public OpenShift IP set to: ${PUBLIC_OPENSHIFT_IP}"
echo

echo "===> Configuring OpenShift:"
docker run --privileged -v ${OPENSHIFT_CONFIG}:/config openshift/origin start master --write-config=/config --kubeconfig=/config/kubeconfig --master=https://localhost:8443 --public-master=https://${PUBLIC_OPENSHIFT_IP}:8443 --etcd=http://etcd:2379
sudo -E chown -R ${USER} ${OPENSHIFT_CONFIG}

# The following assumes GCE and that KUBE_GCE_INSTANCE_PREFIX is set
export ZONE=$(gcloud compute instances list | grep "${KUBE_GCE_INSTANCE_PREFIX}\-master" | awk '{print $2}' | head -1)
echo "sudo cat /srv/kubernetes/server.key; exit;" | gcloud compute ssh ${KUBE_GCE_INSTANCE_PREFIX}-master --zone ${ZONE} | grep -Ex "(^\-.*\-$|^\S+$)" > ${OPENSHIFT_CONFIG}/serviceaccounts.private.key
# The following insertion will fail if indentation changes
sed -i -e 's/publicKeyFiles:.*$/publicKeyFiles:/g' -e '/publicKeyFiles:/a \ \ - serviceaccounts.private.key' ${OPENSHIFT_CONFIG}/master-config.yaml

docker run -it --privileged -e="KUBECONFIG=/config/admin.kubeconfig" -v ${OPENSHIFT_CONFIG}:/config openshift/origin cli secrets new openshift-config /config -o json &> ${OPENSHIFT_EXAMPLE}/secret.json
kubectl create -f ${OPENSHIFT_EXAMPLE}/secret.json --namespace='openshift-origin'
echo

echo "===> Running OpenShift Master:"
kubectl create -f ${OPENSHIFT_EXAMPLE}/openshift-controller.yaml --namespace='openshift-origin'
echo

echo Done.
