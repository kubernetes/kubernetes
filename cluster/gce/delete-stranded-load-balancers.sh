#!/usr/bin/env bash

# Copyright 2015 The Kubernetes Authors.
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

# A utility for deleting target pools and forwarding rules that are unattached to VMs
PROJECT=${PROJECT:-kubernetes-jenkins}
REGION=${REGION:-us-central1}

LIST=$(gcloud --project=${PROJECT} compute target-pools list --format='value(name)')

result=0
for x in ${LIST}; do
    if ! gcloud compute --project=${PROJECT} target-pools get-health "${x}" --region=${REGION} 2>/dev/null >/dev/null; then
	echo DELETING "${x}"
	gcloud compute --project=${PROJECT} firewall-rules delete "k8s-fw-${x}" -q
	gcloud compute --project=${PROJECT} forwarding-rules delete "${x}" --region=${REGION} -q
	gcloud compute --project=${PROJECT} addresses delete "${x}" --region=${REGION} -q
	gcloud compute --project=${PROJECT} target-pools delete "${x}" --region=${REGION} -q
        result=1
    fi
done
exit ${result}

