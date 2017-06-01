package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["services.go"],
    tags = ["automanaged"],
    deps = ["//pkg/kubelet/api/v1alpha1/runtime:go_default_library"],
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
        "//pkg/kubelet/api/testing:all-srcs",
        "//pkg/kubelet/api/v1alpha1/runtime:all-srcs",
        "//pkg/kubelet/api/v1alpha1/stats:all-srcs",
    ],
    tags = ["automanaged"],
)
