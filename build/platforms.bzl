# Copyright 2019 The Kubernetes Authors.
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

# KUBE_SERVER_PLATFORMS in hack/lib/golang.sh
SERVER_PLATFORMS = {
    "linux": [
        "amd64",
        "arm",
        "arm64",
        "ppc64le",
        "s390x",
    ],
}

# KUBE_NODE_PLATFORMS in hack/lib/golang.sh
NODE_PLATFORMS = {
    "linux": [
        "amd64",
        "arm",
        "arm64",
        "ppc64le",
        "s390x",
    ],
    "windows": [
        "amd64",
    ],
}

# KUBE_CLIENT_PLATFORMS in hack/lib/golang.sh
CLIENT_PLATFORMS = {
    "linux": [
        "386",
        "amd64",
        "arm",
        "arm64",
        "ppc64le",
        "s390x",
    ],
    "darwin": [
        "386",
        "amd64",
    ],
    "windows": [
        "386",
        "amd64",
    ],
}

# KUBE_TEST_PLATFORMS in hack/lib/golang.sh
TEST_PLATFORMS = {
    "linux": [
        "amd64",
        "arm",
        "arm64",
        "s390x",
        "ppc64le",
    ],
    "darwin": [
        "amd64",
    ],
    "windows": [
        "amd64",
    ],
}
