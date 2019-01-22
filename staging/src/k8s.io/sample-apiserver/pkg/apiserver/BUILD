package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["scheme_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/apitesting/roundtrip:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/apis/wardle/fuzzer:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = ["apiserver.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/sample-apiserver/pkg/apiserver",
    importpath = "k8s.io/sample-apiserver/pkg/apiserver",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/version:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/registry/rest:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/apis/wardle:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/apis/wardle/install:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/registry:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/registry/wardle/fischer:go_default_library",
        "//staging/src/k8s.io/sample-apiserver/pkg/registry/wardle/flunder:go_default_library",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
