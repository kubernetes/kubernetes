# Copyright 2017 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Makefile for building Go files from protos.

# Change these to match your environment.
PROTOC=$(HOME)/bin/protoc
PROTOC_GO_PLUGIN_DIR=$(GOPATH)/bin

# The repo github.com/google/protobuf.
PROTOBUF_REPO=$(HOME)/git-repos/google/protobuf

gen-protos:  sync-protobuf
	for d in proto/*; do \
	    PATH=$(PATH):$(PROTOC_GO_PLUGIN_DIR) \
	    $(PROTOC) --go_out=plugins=grpc:$$d \
	    -I $$d -I $(PROTOBUF_REPO)/src $$d/*.proto; \
	done


sync-protobuf:
	cd $(PROTOBUF_REPO); git pull

