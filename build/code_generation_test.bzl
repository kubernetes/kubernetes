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

load(":code_generation.bzl", "bazel_go_library", "go_pkg", code_generation = "exported_for_unit_testing")
load("@bazel_skylib//:lib.bzl", "asserts", "unittest")
load("@io_k8s_code_generation//:tags.bzl", "tags_pkgs_values", "tags_values_pkgs")

def _format_output_dict_test_impl(ctx):
    env = unittest.begin(ctx)

    tags_pkgs_values = {
        "openapi-gen": {
            "foo/pkg": {
                "True": None,
            },
            "bar/pkg": {
                "True": None,
                "False": None,
                "Maybe": None,
            },
        },
        "bazel-gen": {
            "build": {
                "Magic": None,
            },
        },
        "don't-escape-me": {
            'but-do-escape-"-me': {
            },
        },
    }

    expected_output = """existing_line = []
things = {
    "bazel-gen": {
        "build": [
            "Magic",
        ],
    },
    "don't-escape-me": {
        "but-do-escape-\\"-me": [
        ],
    },
    "openapi-gen": {
        "bar/pkg": [
            "False",
            "Maybe",
            "True",
        ],
        "foo/pkg": [
            "True",
        ],
    },
}
"""

    output = ["existing_line = []"]
    code_generation._format_output_dict("things", tags_pkgs_values, output)

    asserts.equals(
        env,
        expected_output,
        "\n".join(output),
    )

    unittest.end(env)

format_output_dict_test = unittest.make(_format_output_dict_test_impl)

def _line_to_pkg_tag_values_test_impl(ctx):
    env = unittest.begin(ctx)

    # simple case: single value, no extra :s in tag name
    asserts.equals(
        env,
        struct(
            pkg = "pkg/kubelet/apis/kubeletconfig/v1beta1",
            tag = "openapi-gen",
            values = ["true"],
        ),
        code_generation._line_to_pkg_tag_values(
            ".",
            "./pkg/kubelet/apis/kubeletconfig/v1beta1/doc.go:// +k8s:openapi-gen=true",
        ),
    )

    # : in tag name handled correctly?
    asserts.equals(
        env,
        struct(
            pkg = "pkg/kubectl/cmd/testing",
            tag = "deepcopy-gen:interfaces",
            values = ["k8s.io/apimachinery/pkg/runtime.Object"],
        ),
        code_generation._line_to_pkg_tag_values(
            "/",
            "/pkg/kubectl/cmd/testing/fake.go:// +k8s:deepcopy-gen:interfaces=k8s.io/apimachinery/pkg/runtime.Object",
        ),
    )

    # multiple values handled correctly?
    asserts.equals(
        env,
        struct(
            pkg = "staging/src/k8s.io/api/storage/v1alpha1",
            tag = "deepcopy-gen",
            values = ["package", "register"],
        ),
        code_generation._line_to_pkg_tag_values(
            "/a/path",
            "/a/path/staging/src/k8s.io/api/storage/v1alpha1/doc.go:// +k8s:deepcopy-gen=package,register",
        ),
    )

    # tags in _examples directories should be skipped
    asserts.equals(
        env,
        None,
        code_generation._line_to_pkg_tag_values(
            "/something/else",
            "/something/else/staging/src/k8s.io/code-generator/_examples/apiserver/apis/example/v1/doc.go:// +k8s:defaulter-gen=TypeMeta",
        ),
    )

    unittest.end(env)

line_to_pkg_tag_values_test = unittest.make(_line_to_pkg_tag_values_test_impl)

def _code_generation_tags_bzl_smoke_test_impl(ctx):
    env = unittest.begin(ctx)

    # A few basic tags we expect to exist
    expected_tags = [
        "conversion-gen",
        "deepcopy-gen",
        "defaulter-gen",
        "openapi-gen",
    ]

    for tag in expected_tags:
        asserts.true(
            env,
            tags_pkgs_values.get(tag),
            "Expected to find tag %s in tags_pkgs_values dict" % tag,
        )
        asserts.equals(
            env,
            "dict",
            type(tags_pkgs_values.get(tag)),
        )

        asserts.true(
            env,
            tags_values_pkgs.get(tag),
            "Expected to find tag %s in tags_values_pkgs dict" % tag,
        )
        asserts.equals(
            env,
            "dict",
            type(tags_values_pkgs.get(tag)),
        )

    # Verify structure of the dictionaries
    asserts.equals(
        env,
        "dict",
        type(tags_pkgs_values),
    )
    for tag in tags_pkgs_values.keys():
        asserts.equals(
            env,
            "dict",
            type(tags_pkgs_values[tag]),
        )
        for pkg in tags_pkgs_values[tag]:
            asserts.equals(
                env,
                "list",
                type(tags_pkgs_values[tag][pkg]),
            )

    asserts.equals(
        env,
        "dict",
        type(tags_values_pkgs),
    )
    for tag in tags_values_pkgs.keys():
        asserts.equals(
            env,
            "dict",
            type(tags_values_pkgs[tag]),
        )
        for pkg in tags_values_pkgs[tag]:
            asserts.equals(
                env,
                "list",
                type(tags_values_pkgs[tag][pkg]),
            )

    # Verify some expected values using the openapi-gen tag
    asserts.equals(
        env,
        ["true"],
        tags_pkgs_values.get("openapi-gen", {}).get("pkg/version"),
        "Expected to find openapi-gen=true for pkg/version in tags_pkgs_values",
    )

    asserts.true(
        env,
        "pkg/version" in tags_values_pkgs.get("openapi-gen", {}).get("true", []),
        "Expected to find pkg/version for openapi-gen=true in tags_values_pkgs",
    )

    unittest.end(env)

code_generation_tags_bzl_smoke_test = unittest.make(_code_generation_tags_bzl_smoke_test_impl)

def _bazel_go_library_test_impl(ctx):
    env = unittest.begin(ctx)

    test_cases = [
        ("pkg/kubectl/util", "//pkg/kubectl/util:go_default_library"),
        ("vendor/some/third/party", "//vendor/some/third/party:go_default_library"),
        ("staging/src/k8s.io/apimachinery/api", "//staging/src/k8s.io/apimachinery/api:go_default_library"),
    ]
    for input, expected in test_cases:
        asserts.equals(env, expected, bazel_go_library(input))

    unittest.end(env)

bazel_go_library_test = unittest.make(_bazel_go_library_test_impl)

def _go_pkg_test_impl(ctx):
    env = unittest.begin(ctx)

    test_cases = [
        ("pkg/kubectl/util", "k8s.io/kubernetes/pkg/kubectl/util"),
        ("vendor/some/third/party", "k8s.io/kubernetes/vendor/some/third/party"),
        ("staging/src/k8s.io/apimachinery/api", "k8s.io/kubernetes/vendor/k8s.io/apimachinery/api"),
    ]
    for input, expected in test_cases:
        asserts.equals(env, expected, go_pkg(input))

    unittest.end(env)

go_pkg_test = unittest.make(_go_pkg_test_impl)

def code_generation_test_suite():
    unittest.suite(
        "code_generation_tests",
        line_to_pkg_tag_values_test,
        format_output_dict_test,
        code_generation_tags_bzl_smoke_test,
        bazel_go_library_test,
        go_pkg_test,
    )
