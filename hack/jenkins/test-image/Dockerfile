# Copyright 2015 The Kubernetes Authors All rights reserved.
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

# This file creates a build environment for building and running kubernetes
# unit and integration tests

FROM golang:1.6
MAINTAINER  Jeff Lowdermilk <jeffml@google.com>

# Install pre-requisites
# TODO(fejta): only use valid packages
# file is used when uploading test artifacts to GCS.
# libapparmor1 is needed for docker-in-docker.
# netcat is used by integration test scripts.
# jq is used by hack/verify-godep-licenses.sh
# TODO(fejta): why do we need rsync?
RUN apt-get -o Acquire::Check-Valid-Until=false update \
    && apt-get install -y file libapparmor1 netcat-openbsd jq rsync

# Setup jenkins user, name must be non-root
RUN useradd -m -d /workspace -s /bin/bash jenkins
USER jenkins
WORKDIR /workspace
ENV USER=jenkins \
    WORKSPACE=/workspace \
    TERM=xterm

# Setup the kubernetes go depot
RUN mkdir -p /go/src/k8s.io/kubernetes \
    && ln -s /go/src/k8s.io/kubernetes /workspace/kubernetes

# Install a base version of gcloud
ADD https://dl.google.com/dl/cloudsdk/channels/rapid/google-cloud-sdk.tar.gz /workspace
RUN tar xzf google-cloud-sdk.tar.gz \
    && /workspace/google-cloud-sdk/install.sh \
         --disable-installation-options \
	 --bash-completion=false \
	 --path-update=false \
	 --usage-reporting=false \
    && rm /workspace/google-cloud-sdk.tar.gz
ENV PATH=/workspace/google-cloud-sdk/bin:${PATH} \
    CLOUDSDK_CORE_DISABLE_PROMPTS=1 \
    CLOUDSDK_COMPONENT_MANAGER_DISABLE_UPDATE_CHECK=true \
    CLOUDSDK_CORE_PRINT_UNHANDLED_TRACEBACKS=1

# Attach here a secret volume with private-key and public-key
USER root
RUN mkdir -p /var/lib/jenkins/gce_keys \
    && chown -R jenkins /var/lib/jenkins
USER jenkins

# Defaults for all e2e runs
ENV E2E_UP=true \
    E2E_TEST=true \
    E2E_DOWN=true

# Things you want to customize
ENV E2E_PUBLISH_GREEN_VERSION=false \
    GINKGO_PARALLEL_NODES=25 \
    GINKGO_TEST_ARGS="--ginkgo.focus=\[Conformance\]" \
    INSTANCE_PREFIX=jenkins-e2e \
    JENKINS_GCS_LOGS_PATH=gs://fejternetes/logs \
    JENKINS_UPLOAD_TO_GCS=yes

# GCP specific things
ENV FAIL_ON_GCP_RESOURCE_LEAK=true \
    JOB_NAME=kubernetes-e2e-gce-conformance \
    KUBE_GCE_INSTANCE_PREFIX=jenkins-e2e \
    KUBE_GCE_NETWORK=jenkins-e2e \
    KUBE_GCE_ZONE=us-central1-f \
    KUBERNETES_PROVIDER=gce \
    PROJECT=unset-project

ENTRYPOINT ["/bin/bash", "-c"]
CMD ["set -eux
      && cp /var/lib/jenkins/gce_keys/{private-key,google_compute_engine}
      && cp /var/lib/jenkins/gce_keys/{public-key,google_compute_engine.pub}
      && chown $(whoami) /var/lib/jenkins/gce_keys/google*
      && chmod 600 /var/lib/jenkins/gce_keys/google_compute_engine
      && env
      && bash <(curl -fsS --retry 3 'https://raw.githubusercontent.com/kubernetes/kubernetes/master/hack/jenkins/e2e-runner.sh')
      && echo passed || echo failed"]
