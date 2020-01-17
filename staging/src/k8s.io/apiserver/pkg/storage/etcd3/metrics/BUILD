package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["metrics.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/storage/etcd3/metrics",
    importpath = "k8s.io/apiserver/pkg/storage/etcd3/metrics",
    deps = [
        "//staging/src/k8s.io/component-base/metrics:go_default_library",
        "//staging/src/k8s.io/component-base/metrics/legacyregistry:go_default_library",
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
