# Copyright 2023 The Kubernetes Authors.
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

FROM golang:1.20.1-bullseye as builder

WORKDIR /workspace

# Copy the source
COPY apimachinery/ apimachinery/
COPY kms/ kms/
WORKDIR /workspace/kms/internal/plugins/base64

ARG TARGETARCH
ARG TARGETPLATFORM
RUN CGO_ENABLED=0 GOOS=linux GOARCH=${TARGETARCH} GO111MODULE=on go build -a -o base64-plugin plugin.go
RUN chmod +x base64-plugin

# Use distroless as minimal base image to package the manager binary
# Refer to https://github.com/GoogleContainerTools/distroless for more details
FROM --platform=${TARGETPLATFORM:-linux/amd64} gcr.io/distroless/static:nonroot
WORKDIR /
COPY --from=builder /workspace/kms/internal/plugins/base64/base64-plugin .

ENTRYPOINT [ "/base64-plugin" ]
