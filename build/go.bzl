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

# Defines several go_binary rules to work around a Bazel issue which makes
# the pure attribute on go_binary not configurable.
# The name provided will have cgo enabled if targeting Linux, and will
# be a pure go binary otherwise. Additionally, if targeting Windows, the
# output filename will have a .exe suffix.
def go_binary_conditional_pure(name, tags = None, **kwargs):
    tags = tags or []
    tags.append("manual")
    go_binary(
        name = "_%s-cgo" % name,
        out = name,
        pure = "off",
        tags = tags,
        **kwargs
    )

    # Define a rule for both Unix and Windows exe suffixes.
    [go_binary(
        name = "_%s-pure" % out,
        out = out,
        pure = "on",
        tags = tags,
        **kwargs
    ) for out in [name, name + ".exe"]]

    # The real magic, where we work around the pure attribute not being
    # configurable: select the appropriate go_binary rule above based on the
    # configured platform.
    native.alias(
        name = name,
        actual = select({
            "@io_bazel_rules_go//go/platform:linux": ":_%s-cgo" % name,
            "@io_bazel_rules_go//go/platform:windows": ":_%s.exe-pure" % name,
            "//conditions:default": ":_%s-pure" % name,
        }),
    )

# Defines several go_test rules to work around a Bazel issue which makes
# the pure attribute on go_test not configurable.
# This also defines genrules to produce test binaries named ${out} and
# ${out}.exe, and an alias named ${out}_binary which automatically selects
# the correct filename suffix (i.e. with a .exe on Windows).
def go_test_conditional_pure(name, out, tags = None, **kwargs):
    tags = tags or []
    tags.append("manual")

    go_test(
        name = "_%s-cgo" % name,
        pure = "off",
        testonly = False,
        tags = tags,
        **kwargs
    )

    go_test(
        name = "_%s-pure" % name,
        pure = "on",
        testonly = False,
        tags = tags,
        **kwargs
    )

    native.alias(
        name = name,
        actual = select({
            "@io_bazel_rules_go//go/platform:linux": ":_%s-cgo" % name,
            "//conditions:default": ":_%s-pure" % name,
        }),
    )

    [native.genrule(
        name = "gen_%s" % o,
        srcs = [name],
        outs = [o],
        cmd = "cp $< $@;",
        output_to_bindir = True,
        executable = True,
        tags = tags,
    ) for o in [out, out + ".exe"]]

    native.alias(
        name = "%s_binary" % out,
        actual = select({
            "@io_bazel_rules_go//go/platform:windows": ":gen_%s.exe" % out,
            "//conditions:default": ":gen_%s" % out,
        }),
    )
