# Copyright 2016 The Kubernetes Authors.
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

# This file creates a standard build environment for building Kubernetes
FROM gcr.io/google_containers/kube-cross:KUBE_BUILD_IMAGE_CROSS_TAG

# Mark this as a kube-build container
RUN touch /kube-build-image

# TO run as non-root we sometimes need to rebuild go stdlib packages.
RUN chmod -R a+rwx /usr/local/go/pkg

# The kubernetes source is expected to be mounted here.  This will be the base
# of operations.
ENV HOME /go/src/k8s.io/kubernetes
WORKDIR ${HOME}
# We have to mkdir the dirs we need, or else Docker will create them when we
# mount volumes, and it will create them with root-only permissions.  The
# explicit chmod of _output is required, but I can't really explain why.
RUN mkdir -p ${HOME} ${HOME}/_output \
    && chmod -R a+rwx ${HOME} ${HOME}/_output

# Propagate the git tree version into the build image
ADD kube-version-defs /kube-version-defs
RUN chmod a+r /kube-version-defs
ENV KUBE_GIT_VERSION_FILE /kube-version-defs

# Make output from the dockerized build go someplace else
ENV KUBE_OUTPUT_SUBPATH _output/dockerized

# Upload Kubernetes source
ADD kube-source.tar.gz /go/src/k8s.io/kubernetes/
