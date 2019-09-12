load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["default_registry.go"],
    importpath = "k8s.io/kubernetes/pkg/scheduler/framework/plugins",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/scheduler/framework/plugins/noop:go_default_library",
        "//pkg/scheduler/framework/v1alpha1:go_default_library",
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
        "//pkg/scheduler/framework/plugins/examples:all-srcs",
        "//pkg/scheduler/framework/plugins/noop:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
