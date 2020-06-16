package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["kube_features.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/controller-manager/pkg/features",
    importpath = "k8s.io/controller-manager/pkg/features",
    deps = ["//staging/src/k8s.io/component-base/featuregate:go_default_library"],
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
        "//staging/src/k8s.io/controller-manager/pkg/features/register:all-srcs",
    ],
    tags = ["automanaged"],
)
