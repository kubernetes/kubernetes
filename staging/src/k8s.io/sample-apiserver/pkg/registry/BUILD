package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["registry.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/sample-apiserver/pkg/registry",
    importpath = "k8s.io/sample-apiserver/pkg/registry",
    deps = [
        "//staging/src/k8s.io/apiserver/pkg/registry/generic/registry:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/registry/rest:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//staging/src/k8s.io/sample-apiserver/pkg/registry/wardle/fischer:all-srcs",
        "//staging/src/k8s.io/sample-apiserver/pkg/registry/wardle/flunder:all-srcs",
    ],
    tags = ["automanaged"],
)
