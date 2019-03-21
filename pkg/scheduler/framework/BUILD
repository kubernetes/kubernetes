load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["registrar.go"],
    importpath = "k8s.io/kubernetes/pkg/scheduler/plugins",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/scheduler/internal/cache:go_default_library",
        "//pkg/scheduler/plugins/v1alpha1:go_default_library",
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
        "//pkg/scheduler/plugins/examples:all-srcs",
        "//pkg/scheduler/plugins/v1alpha1:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
