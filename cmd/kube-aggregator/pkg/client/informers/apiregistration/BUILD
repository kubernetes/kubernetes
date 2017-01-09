package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["interface.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/kube-aggregator/pkg/client/informers/apiregistration/internalversion:go_default_library",
        "//cmd/kube-aggregator/pkg/client/informers/apiregistration/v1alpha1:go_default_library",
        "//cmd/kube-aggregator/pkg/client/informers/internalinterfaces:go_default_library",
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
        "//cmd/kube-aggregator/pkg/client/informers/apiregistration/internalversion:all-srcs",
        "//cmd/kube-aggregator/pkg/client/informers/apiregistration/v1alpha1:all-srcs",
    ],
    tags = ["automanaged"],
)
