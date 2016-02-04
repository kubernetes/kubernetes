#!/bin/bash

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

# Build etcd from source in a docker container

# Require three arguments
if [[ $# != 3 ]]; then
	echo "Usage: ./build-etcd.sh TAG ARCH TARGET_DIR"
	exit
fi

# Example tag should be 2.2.1, not v2.2.1
TAG=$1
ARCH=$2
TARGET_DIR=$3

GOLANG_VERSION=${GOLANG_VERSION:-1.5.3}
GOARM=6

# Create the ${TARGET_DIR} directory, if it doesn't exist
mkdir -p ${TARGET_DIR}

# Do not compile if we should make an image for amd64, use the "official" etcd binaries instead
if [[ ${ARCH} == "amd64" ]]; then

	# Just download the binaries to ${TARGET_DIR}
	curl -sSL https://github.com/coreos/etcd/releases/download/v${TAG}/etcd-v${TAG}-linux-amd64.tar.gz | tar -xz -C ${TARGET_DIR} --strip-components=1
else
	# Download etcd in a golang container and cross-compile it statically
	CID=$(docker run -d golang:${GOLANG_VERSION} /bin/sh -c \
		"mkdir /etcd \
		&& curl -sSL https://github.com/coreos/etcd/archive/v${TAG}.tar.gz | tar -C /etcd -xz --strip-components=1 \
		&& cd /etcd \
		&& GOARM=${GOARM} GOARCH=${ARCH} ./build")
	
	# Wait until etcd has compiled
	docker wait ${CID}

	# Copy out the bin folder to ${TARGET_DIR}
	docker cp ${CID}:/etcd/bin ${TARGET_DIR}

	# Move the contents in bin to the target directory
	mv ${TARGET_DIR}/bin/* ${TARGET_DIR}
fi
