load("@io_bazel_rules_go//go:def.bzl", "go_library")

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
        "//pkg/scheduler/framework/v1alpha1:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

go_library(
    name = "go_default_library",
    srcs = ["register.go"],
    importpath = "k8s.io/kubernetes/pkg/scheduler/framework",
    visibility = ["//visibility:public"],
    deps = ["//pkg/scheduler/framework/v1alpha1:go_default_library"],
)
