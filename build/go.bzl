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

load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_context", "go_test")
load("@io_bazel_rules_go//go/platform:list.bzl", "GOOS_GOARCH")

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

_GO_BUILD_MODE_TMPL = "{goos}/{goarch}/pure={pure},static={static},msan={msan},race={race}\n"

def _go_build_mode_aspect_impl(target, ctx):
    if (not hasattr(ctx.rule.attr, "_is_executable") or
        not ctx.rule.attr._is_executable or
        ctx.rule.attr.testonly):
        # We only care about exporting platform info for executable targets
        # that aren't testonly (e.g. kubectl and e2e.test).
        return []

    mode = go_context(ctx).mode

    out = ctx.actions.declare_file(
        target.files_to_run.executable.basename + ".go_build_mode",
        sibling = target.files_to_run.executable,
    )
    ctx.actions.write(out, _GO_BUILD_MODE_TMPL.format(
        goos = mode.goos,
        goarch = mode.goarch,
        pure = str(mode.pure).lower(),
        static = str(mode.static).lower(),
        msan = str(mode.msan).lower(),
        race = str(mode.race).lower(),
    ))

    return [OutputGroupInfo(default = depset([out]))]

# This aspect ouputs a *.go_build_mode metadata for go binaries. This metadata
# is used for executable selection e.g. in CI.
go_build_mode_aspect = aspect(
    implementation = _go_build_mode_aspect_impl,
    attrs = {
        "goos": attr.string(
            default = "auto",
            values = ["auto"] + {goos: None for goos, _ in GOOS_GOARCH}.keys(),
        ),
        "goarch": attr.string(
            default = "auto",
            values = ["auto"] + {goarch: None for _, goarch in GOOS_GOARCH}.keys(),
        ),
        "pure": attr.string(
            default = "auto",
            values = ["auto", "on", "off"],
        ),
        "static": attr.string(
            default = "auto",
            values = ["auto", "on", "off"],
        ),
        "msan": attr.string(
            default = "auto",
            values = ["auto", "on", "off"],
        ),
        "race": attr.string(
            default = "auto",
            values = ["auto", "on", "off"],
        ),
        "_go_context_data": attr.label(default = "@io_bazel_rules_go//:go_context_data"),
    },
    toolchains = ["@io_bazel_rules_go//go:toolchain"],
)
