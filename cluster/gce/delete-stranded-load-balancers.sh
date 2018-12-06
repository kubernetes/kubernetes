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

# A utility for deleting stranded load balancer resources in GCE
PROJECT=${PROJECT:-kubernetes-jenkins}
REGION=${REGION:-us-central1}

# Deleting external load balancer resources
LIST=$(gcloud --project=${PROJECT} compute target-pools list --format='value(name)')
for x in ${LIST}; do
  # Check the existence of vm instance to see if the load balancer resources are
  # actively used. Only delete them if they are not.
  if ! gcloud compute --project=${PROJECT} target-pools get-health "${x}" --region=${REGION} 2>/dev/null >/dev/null; then
    echo DELETING LB "${x}"
    gcloud compute --project=${PROJECT} firewall-rules delete "k8s-fw-${x}" -q
    gcloud compute --project=${PROJECT} forwarding-rules delete "${x}" --region=${REGION} -q
    gcloud compute --project=${PROJECT} addresses delete "${x}" --region=${REGION} -q
    gcloud compute --project=${PROJECT} target-pools delete "${x}" --region=${REGION} -q
  fi
done

# Deleting internal load balancer resources
ILB_LIST=$(gcloud --project=${PROJECT} compute backend-services list --format='value(name)')
for x in ${ILB_LIST}; do
  # Check the existence of vm instance to see if the load balancer resources are
  # actively used. Only delete them if they are not.
  if ! gcloud compute --project=${PROJECT} backend-services get-health "${x}" --region=${REGION} 2>/dev/null >/dev/null; then
    echo DELETING internal LB "${x}"
    ig=$(gcloud compute --project=${PROJECT} backend-services list --regions=${REGION} 2>/dev/null | grep "${x}" | awk '{print $2}' | cut -d'/' -f3)
    zone=$(gcloud compute --project=${PROJECT} instance-groups unmanaged list | grep ${ig} | awk '{print $2}')
    gcloud compute --project=${PROJECT} firewall-rules delete "${x}" -q
    gcloud compute --project=${PROJECT} forwarding-rules delete "${x}" --region=${REGION} -q
    gcloud compute --project=${PROJECT} backend-services delete "${x}" --region=${REGION} -q
    gcloud compute --project=${PROJECT} instance-groups unmanaged delete "${ig}" --zone=${zone} -q
  fi
done

