package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["fuzzer.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apimachinery/pkg/apis/meta/fuzzer",
    importpath = "k8s.io/apimachinery/pkg/apis/meta/fuzzer",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/apitesting:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/apitesting/fuzzer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/github.com/google/gofuzz:go_default_library",
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
