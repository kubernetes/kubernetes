load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["volume_binding.go"],
    importpath = "k8s.io/kubernetes/pkg/scheduler/framework/plugins/volumebinding",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/scheduler/algorithm/predicates:go_default_library",
        "//pkg/scheduler/framework/plugins/migration:go_default_library",
        "//pkg/scheduler/framework/v1alpha1:go_default_library",
        "//pkg/scheduler/nodeinfo:go_default_library",
        "//pkg/scheduler/volumebinder:go_default_library",
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
