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

def go_platform_constraint(os, arch):
    return "@io_bazel_rules_go//go/platform:%s_%s" % (os, arch)

def _update_dict_for_platform_category(d, value, platforms, only_os = None):
    if not value:
        return
    for os, arches in platforms.items():
        if only_os and os != only_os:
            continue
        for arch in arches:
            constraint = go_platform_constraint(os, arch)
            if type(value) == "list":
                d.setdefault(constraint, []).extend(
                    [v.format(OS = os, ARCH = arch) for v in value],
                )
            else:
                if constraint in d:
                    fail("duplicate entry for constraint %s", constraint)
                d[constraint] = value.format(OS = os, ARCH = arch)

def for_platforms(
        for_client = None,
        for_node = None,
        for_server = None,
        for_test = None,
        default = None,
        only_os = None):
    d = {}
    if default != None:
        d["//conditions:default"] = default
    _update_dict_for_platform_category(d, for_client, CLIENT_PLATFORMS, only_os)
    _update_dict_for_platform_category(d, for_node, NODE_PLATFORMS, only_os)
    _update_dict_for_platform_category(d, for_server, SERVER_PLATFORMS, only_os)
    _update_dict_for_platform_category(d, for_test, TEST_PLATFORMS, only_os)
    return d
