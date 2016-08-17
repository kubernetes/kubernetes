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

DBG_MAKEFILE ?=
ifeq ($(DBG_MAKEFILE),1)
    $(warning ***** starting makefile for goal(s) "$(MAKECMDGOALS)")
    $(warning ***** $(shell date))
else
    # If we're not debugging the Makefile, don't echo recipes.
    MAKEFLAGS += -s
endif

.PHONY: all
all: init build push deploy

.PHONY: init
init:
	./develop/develop.sh init

.PHONY: build
build: build_binaries build_image

.PHONY: push
push:
	./develop/develop.sh push

.PHONY: deploy
deploy: deploy_clusters deploy_federation

.PHONY: destroy
destroy: destroy_federation destroy_clusters

.PHONY: build_binaries
build_binaries:
	./develop/develop.sh build_binaries

.PHONY: build_image
build_image:
	./develop/develop.sh build_image

.PHONY: deploy_clusters
deploy_clusters:
	./develop/develop.sh deploy_clusters

.PHONY: deploy_federation
deploy_federation:
	./develop/develop.sh deploy_federation

.PHONY: destroy_federation
destroy_federation:
	./develop/develop.sh destroy_federation

.PHONY: destroy_clusters
destroy_clusters:
	./develop/develop.sh destroy_clusters

.PHONY: redeploy_federation
redeploy_federation:
	./develop/develop.sh redeploy_federation
