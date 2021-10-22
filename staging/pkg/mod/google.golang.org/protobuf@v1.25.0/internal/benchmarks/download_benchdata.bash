#!/bin/bash
# Copyright 2019 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

cd "$(git rev-parse --show-toplevel)"
mkdir -p .cache/benchdata
cd .cache/benchdata

# Download small benchmark datasets.
PROTOBUF_VERSION=v3.11.4
curl -s -O https://raw.githubusercontent.com/protocolbuffers/protobuf/$PROTOBUF_VERSION/benchmarks/datasets/google_message1/proto2/dataset.google_message1_proto2.pb
curl -s -O https://raw.githubusercontent.com/protocolbuffers/protobuf/$PROTOBUF_VERSION/benchmarks/datasets/google_message1/proto3/dataset.google_message1_proto3.pb
curl -s -O https://raw.githubusercontent.com/protocolbuffers/protobuf/$PROTOBUF_VERSION/benchmarks/datasets/google_message2/dataset.google_message2.pb

# Download large benchmark datasets.
curl -s https://storage.googleapis.com/protobuf_opensource_benchmark_data/datasets.tar.gz | tar zx
