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

# Build the etcd image
#
# Usage:
# 	[TAGS=2.2.1 2.3.7 3.0.17 3.1.12] [REGISTRY=k8s.gcr.io] [ARCH=amd64] [BASEIMAGE=busybox] make (build|push)

# The image contains different etcd versions to simplify
# upgrades. Thus be careful when removing any tag from here.
#
# NOTE: The etcd upgrade rules are that you can upgrade only 1 minor
# version at a time, and patch release don't matter.
#
# Except from etcd-$(tag) and etcdctl-$(tag) binaries, we also
# need etcd and etcdctl binaries for backward compatibility reasons.
# That binary will be set to the last tag from $(TAGS).
TAGS?=2.2.1 2.3.7 3.0.17 3.1.12
REGISTRY_TAG?=3.1.12
# ROLLBACK_REGISTRY_TAG specified the tag that REGISTRY_TAG may be rolled back to.
ROLLBACK_REGISTRY_TAG?=3.1.12
ARCH?=amd64
# Image should be pulled from k8s.gcr.io, which will auto-detect
# region (us, eu, asia, ...) and pull from the closest.
REGISTRY?=k8s.gcr.io
# Images should be pushed to staging-k8s.gcr.io.
PUSH_REGISTRY?=staging-k8s.gcr.io
# golang version should match the golang version from https://github.com/coreos/etcd/releases for REGISTRY_TAG version of etcd.
GOLANG_VERSION?=1.8.5
GOARM=7
TEMP_DIR:=$(shell mktemp -d)

ifeq ($(ARCH),amd64)
	BASEIMAGE?=busybox
endif
ifeq ($(ARCH),arm)
	BASEIMAGE?=arm32v7/busybox
endif
ifeq ($(ARCH),arm64)
	BASEIMAGE?=arm64v8/busybox
endif
ifeq ($(ARCH),ppc64le)
	BASEIMAGE?=ppc64le/busybox
endif
ifeq ($(ARCH),s390x)
	BASEIMAGE?=s390x/busybox
endif

build:
	# Copy the content in this dir to the temp dir,
	# without copying the subdirectories.
	find ./ -maxdepth 1 -type f | xargs -I {} cp {} $(TEMP_DIR)

	# Compile attachlease
	docker run --interactive -v $(shell pwd)/../../../:/go/src/k8s.io/kubernetes -v $(TEMP_DIR):/build -e GOARCH=$(ARCH) golang:$(GOLANG_VERSION) \
		/bin/bash -c "CGO_ENABLED=0 go build -o /build/attachlease k8s.io/kubernetes/cluster/images/etcd/attachlease"
	# Compile rollback
	docker run --interactive -v $(shell pwd)/../../../:/go/src/k8s.io/kubernetes -v $(TEMP_DIR):/build -e GOARCH=$(ARCH) golang:$(GOLANG_VERSION) \
		/bin/bash -c "CGO_ENABLED=0 go build -o /build/rollback k8s.io/kubernetes/cluster/images/etcd/rollback"


ifeq ($(ARCH),amd64)

	# Do not compile if we should make an image for amd64, use the official etcd binaries instead
	# For each release create a tmp dir 'etcd_release_tmp_dir' and unpack the release tar there.
	for tag in $(TAGS); do \
		etcd_release_tmp_dir=$(shell mktemp -d); \
		curl -sSL --retry 5 https://github.com/coreos/etcd/releases/download/v$$tag/etcd-v$$tag-linux-amd64.tar.gz | tar -xz -C $$etcd_release_tmp_dir --strip-components=1; \
		cp $$etcd_release_tmp_dir/etcd $$etcd_release_tmp_dir/etcdctl $(TEMP_DIR)/; \
		cp $(TEMP_DIR)/etcd $(TEMP_DIR)/etcd-$$tag; \
		cp $(TEMP_DIR)/etcdctl $(TEMP_DIR)/etcdctl-$$tag; \
	done
else

	# Download etcd in a golang container and cross-compile it statically
	# For each release create a tmp dir 'etcd_release_tmp_dir' and unpack the release tar there.
	for tag in $(TAGS); do \
		etcd_release_tmp_dir=$(shell mktemp -d); \
		docker run --interactive -v $${etcd_release_tmp_dir}:/etcdbin golang:$(GOLANG_VERSION) /bin/bash -c \
			"git clone https://github.com/coreos/etcd /go/src/github.com/coreos/etcd \
			&& cd /go/src/github.com/coreos/etcd \
			&& git checkout v$${tag} \
			&& GOARM=$(GOARM) GOARCH=$(ARCH) ./build \
			&& cp -f bin/$(ARCH)/etcd* bin/etcd* /etcdbin; echo 'done'"; \
		cp $$etcd_release_tmp_dir/etcd $$etcd_release_tmp_dir/etcdctl $(TEMP_DIR)/; \
		cp $(TEMP_DIR)/etcd $(TEMP_DIR)/etcd-$$tag; \
		cp $(TEMP_DIR)/etcdctl $(TEMP_DIR)/etcdctl-$$tag; \
	done

	# Add this ENV variable in order to workaround an unsupported arch blocker
	# The multiarch feature is in an limited and experimental state right now, and etcd should work fine on arm64
	# On arm (which is 32-bit), it can't handle >1GB data in-memory, but it is very unlikely someone tinkering with their limited arm devices would reach such a high usage
	# ppc64le is still quite untested, but compiles and is probably in the process of being validated by IBM.
	cd $(TEMP_DIR) && echo "ENV ETCD_UNSUPPORTED_ARCH=$(ARCH)" >> Dockerfile
endif

	# Replace BASEIMAGE with the real base image
	cd $(TEMP_DIR) && sed -i.bak 's|BASEIMAGE|$(BASEIMAGE)|g' Dockerfile

	# And build the image
	docker build --pull -t $(REGISTRY)/etcd-$(ARCH):$(REGISTRY_TAG) $(TEMP_DIR)

push: build
	docker tag $(REGISTRY)/etcd-$(ARCH):$(REGISTRY_TAG) $(PUSH_REGISTRY)/etcd-$(ARCH):$(REGISTRY_TAG)
	docker push $(PUSH_REGISTRY)/etcd-$(ARCH):$(REGISTRY_TAG)

ifeq ($(ARCH),amd64)
	# Backward compatibility. TODO: deprecate this image tag
	docker tag $(REGISTRY)/etcd-$(ARCH):$(REGISTRY_TAG) $(REGISTRY)/etcd:$(REGISTRY_TAG)
	docker push $(REGISTRY)/etcd:$(REGISTRY_TAG)
endif

ETCD2_ROLLBACK_NEW_TAG=3.0.17
ETCD2_ROLLBACK_OLD_TAG=2.2.1

# Test a rollback to etcd2 from the earliest etcd3 version.
test-rollback-etcd2:
	mkdir -p $(TEMP_DIR)/rollback-etcd2
	cd $(TEMP_DIR)/rollback-etcd2

	@echo "Starting $(ETCD2_ROLLBACK_NEW_TAG) etcd and writing some sample data."
	docker run --tty --interactive -v $(TEMP_DIR)/rollback-etcd2:/var/etcd \
		-e "TARGET_STORAGE=etcd3" \
		-e "TARGET_VERSION=$(ETCD2_ROLLBACK_NEW_TAG)" \
		-e "DATA_DIRECTORY=/var/etcd/data" \
		$(REGISTRY)/etcd-$(ARCH):$(REGISTRY_TAG) /bin/sh -c \
			'INITIAL_CLUSTER=etcd-$$(hostname)=http://localhost:2380 \
			/usr/local/bin/migrate-if-needed.sh && \
			source /usr/local/bin/start-stop-etcd.sh && \
			START_STORAGE=etcd3 START_VERSION=$(ETCD2_ROLLBACK_NEW_TAG) start_etcd && \
			ETCDCTL_API=3 /usr/local/bin/etcdctl-$(ETCD2_ROLLBACK_NEW_TAG) --endpoints http://127.0.0.1:$${ETCD_PORT} put /registry/k1 value1 && \
			stop_etcd && \
			[ $$(cat /var/etcd/data/version.txt) = $(ETCD2_ROLLBACK_NEW_TAG)/etcd3 ]'

	@echo "Rolling back to the previous version of etcd and recording keyspace to a flat file."
	docker run --tty --interactive -v $(TEMP_DIR)/rollback-etcd2:/var/etcd \
		-e "TARGET_STORAGE=etcd2" \
		-e "TARGET_VERSION=$(ETCD2_ROLLBACK_OLD_TAG)" \
		-e "DATA_DIRECTORY=/var/etcd/data" \
		$(REGISTRY)/etcd-$(ARCH):$(REGISTRY_TAG) /bin/sh -c \
			'INITIAL_CLUSTER=etcd-$$(hostname)=http://localhost:2380 \
			/usr/local/bin/migrate-if-needed.sh && \
			source /usr/local/bin/start-stop-etcd.sh && \
			START_STORAGE=etcd2 START_VERSION=$(ETCD2_ROLLBACK_OLD_TAG) start_etcd && \
			/usr/local/bin/etcdctl-$(ETCD2_ROLLBACK_OLD_TAG) --endpoint 127.0.0.1:$${ETCD_PORT} get /registry/k1 > /var/etcd/keyspace.txt && \
			stop_etcd'

	@echo "Checking if rollback successfully downgraded etcd to $(ETCD2_ROLLBACK_OLD_TAG)"
	docker run --tty --interactive -v $(TEMP_DIR)/rollback-etcd2:/var/etcd \
		$(REGISTRY)/etcd-$(ARCH):$(REGISTRY_TAG) /bin/sh -c \
			'[ $$(cat /var/etcd/data/version.txt) = $(ETCD2_ROLLBACK_OLD_TAG)/etcd2 ] && \
			 grep -q value1 /var/etcd/keyspace.txt'

# Test a rollback from the latest version to the previous version.
test-rollback:
	mkdir -p $(TEMP_DIR)/rollback-test
	cd $(TEMP_DIR)/rollback-test

	@echo "Starting $(REGISTRY_TAG) etcd and writing some sample data."
	docker run --tty --interactive -v $(TEMP_DIR)/rollback-test:/var/etcd \
		-e "TARGET_STORAGE=etcd3" \
		-e "TARGET_VERSION=$(REGISTRY_TAG)" \
		-e "DATA_DIRECTORY=/var/etcd/data" \
		$(REGISTRY)/etcd-$(ARCH):$(REGISTRY_TAG) /bin/sh -c \
			'INITIAL_CLUSTER=etcd-$$(hostname)=http://localhost:2380 \
			/usr/local/bin/migrate-if-needed.sh && \
			source /usr/local/bin/start-stop-etcd.sh && \
			START_STORAGE=etcd3 START_VERSION=$(REGISTRY_TAG) start_etcd && \
			ETCDCTL_API=3 /usr/local/bin/etcdctl --endpoints http://127.0.0.1:$${ETCD_PORT} put /registry/k1 value1 && \
			stop_etcd'

	@echo "Rolling back to the previous version of etcd and recording keyspace to a flat file."
	docker run --tty --interactive -v $(TEMP_DIR)/rollback-test:/var/etcd \
		-e "TARGET_STORAGE=etcd3" \
		-e "TARGET_VERSION=$(ROLLBACK_REGISTRY_TAG)" \
		-e "DATA_DIRECTORY=/var/etcd/data" \
		$(REGISTRY)/etcd-$(ARCH):$(REGISTRY_TAG) /bin/sh -c \
			'INITIAL_CLUSTER=etcd-$$(hostname)=http://localhost:2380 \
			/usr/local/bin/migrate-if-needed.sh && \
			source /usr/local/bin/start-stop-etcd.sh && \
			START_STORAGE=etcd3 START_VERSION=$(ROLLBACK_REGISTRY_TAG) start_etcd && \
			ETCDCTL_API=3 /usr/local/bin/etcdctl --endpoints http://127.0.0.1:$${ETCD_PORT} get --prefix / > /var/etcd/keyspace.txt && \
			stop_etcd'

	@echo "Checking if rollback successfully downgraded etcd to $(ROLLBACK_REGISTRY_TAG)"
	docker run --tty --interactive -v $(TEMP_DIR)/rollback-test:/var/etcd \
		$(REGISTRY)/etcd-$(ARCH):$(REGISTRY_TAG) /bin/sh -c \
			'[ $$(cat /var/etcd/data/version.txt) = $(ROLLBACK_REGISTRY_TAG)/etcd3 ] && \
			 grep -q value1 /var/etcd/keyspace.txt'

# Test migrating from each supported versions to the latest version.
test-migrate:
	for tag in $(TAGS); do \
		echo "Testing migration from $${tag} to $(REGISTRY_TAG)" && \
		mkdir -p $(TEMP_DIR)/migrate-$${tag} && \
		cd $(TEMP_DIR)/migrate-$${tag} && \
		MAJOR_VERSION=$$(echo $${tag} | cut -c 1) && \
		echo "Starting etcd $${tag} and writing sample data to keyspace" && \
		docker run --tty --interactive -v $(TEMP_DIR)/migrate-$${tag}:/var/etcd \
			-e "TARGET_STORAGE=etcd$${MAJOR_VERSION}" \
			-e "TARGET_VERSION=$${tag}" \
			-e "DATA_DIRECTORY=/var/etcd/data" \
			$(REGISTRY)/etcd-$(ARCH):$(REGISTRY_TAG) /bin/sh -c \
				"INITIAL_CLUSTER=etcd-\$$(hostname)=http://localhost:2380 \
				/usr/local/bin/migrate-if-needed.sh && \
				source /usr/local/bin/start-stop-etcd.sh && \
				START_STORAGE=etcd$${MAJOR_VERSION} START_VERSION=$${tag} start_etcd && \
				if [ $${MAJOR_VERSION} == 2 ]; then \
				  /usr/local/bin/etcdctl --endpoint http://127.0.0.1:\$${ETCD_PORT} set /registry/k1 value1; \
				else \
				  ETCDCTL_API=3 /usr/local/bin/etcdctl --endpoints http://127.0.0.1:\$${ETCD_PORT} put /registry/k1 value1; \
				fi && \
				stop_etcd" && \
		echo " Migrating from $${tag} to $(REGISTRY_TAG) and capturing keyspace" && \
		docker run --tty --interactive -v $(TEMP_DIR)/migrate-$${tag}:/var/etcd \
			-e "TARGET_STORAGE=etcd3" \
			-e "TARGET_VERSION=$(REGISTRY_TAG)" \
			-e "DATA_DIRECTORY=/var/etcd/data" \
			$(REGISTRY)/etcd-$(ARCH):$(REGISTRY_TAG) /bin/sh -c \
				'INITIAL_CLUSTER=etcd-$$(hostname)=http://localhost:2380 \
				/usr/local/bin/migrate-if-needed.sh && \
				source /usr/local/bin/start-stop-etcd.sh && \
				START_STORAGE=etcd3 START_VERSION=$(REGISTRY_TAG) start_etcd && \
				ETCDCTL_API=3 /usr/local/bin/etcdctl --endpoints http://127.0.0.1:$${ETCD_PORT} get --prefix / > /var/etcd/keyspace.txt && \
				stop_etcd'  && \
		echo "Checking if migrate from $${tag} successfully upgraded etcd to $(REGISTRY_TAG)" && \
		docker run --tty --interactive -v $(TEMP_DIR)/migrate-$${tag}:/var/etcd \
			$(REGISTRY)/etcd-$(ARCH):$(REGISTRY_TAG) /bin/sh -c \
				'[ $$(cat /var/etcd/data/version.txt) = $(REGISTRY_TAG)/etcd3 ] && \
				 grep -q value1 /var/etcd/keyspace.txt'; \
	done

test: test-rollback test-rollback-etcd2 test-migrate

all: build test
.PHONY:	build push test-rollback test-rollback-etcd2 test-migrate test
