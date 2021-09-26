# Copyright 2018 The Kubernetes Authors.
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

.PHONY: verify
verify: verify-fmt verify-lint verify-apidiff vet test

.PHONY: test
test:
	GO111MODULE=on go test -v -race ./...

.PHONY: verify-fmt
verify-fmt:
	GO111MODULE=on ./hack/verify-gofmt.sh

.PHONY: verify-lint
verify-lint:
	GO111MODULE=on ./hack/verify-golint.sh

.PHONY: verify-apidiff
verify-apidiff:
	GO111MODULE=on ./hack/verify-apidiff.sh -r master

.PHONY: vet
vet:
	GO111MODULE=on ./hack/verify-govet.sh

.PHONY: update-fmt
update-fmt:
	gofmt -s -w .
