load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["node_info.go"],
    importpath = "k8s.io/kubernetes/pkg/scheduler/nodeinfo",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/scheduler/framework/v1alpha1:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
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
    visibility = ["//visibility:public"],
)
