package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "deepcopy.go",
        "doc.go",
        "helpers.go",
        "register.go",
        "types.go",
        "types_jsonschema.go",
        "zz_generated.deepcopy.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions",
    importpath = "k8s.io/apiextensions-apiserver/pkg/apis/apiextensions",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["helpers_test.go"],
    embed = [":go_default_library"],
    deps = ["//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library"],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/fuzzer:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/install:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/v1beta1:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions/validation:all-srcs",
    ],
    tags = ["automanaged"],
)
