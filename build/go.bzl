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

load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_test")

# Defines a go_binary rule that enables cgo on platform builds targeting Linux,
# and otherwise builds a pure go binary.
def go_binary_conditional_pure(name, tags = [], **kwargs):
    go_binary(
        name = name,
        pure = select({
            "@io_bazel_rules_go//go/platform:linux": "off",
            "//conditions:default": "on",
        }),
        tags = ["manual"] + tags,
        **kwargs
    )

# Defines a go_test rule that enables cgo on platform builds targeting Linux,
# and otherwise builds a pure go binary.
def go_test_conditional_pure(name, out, tags = [], **kwargs):
    tags.append("manual")

    go_test(
        name = out,
        pure = select({
            "@io_bazel_rules_go//go/platform:linux": "off",
            "//conditions:default": "on",
        }),
        testonly = False,
        tags = tags,
        **kwargs
    )

    native.alias(
        name = "name",
        actual = out,
    )
