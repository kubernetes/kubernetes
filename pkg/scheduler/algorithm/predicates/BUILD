load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["predicates.go"],
    importpath = "k8s.io/kubernetes/pkg/scheduler/algorithm/predicates",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/scheduler/framework/plugins/helper:go_default_library",
        "//pkg/scheduler/framework/plugins/nodeaffinity:go_default_library",
        "//pkg/scheduler/framework/plugins/nodename:go_default_library",
        "//pkg/scheduler/framework/plugins/nodeports:go_default_library",
        "//pkg/scheduler/framework/plugins/noderesources:go_default_library",
        "//pkg/scheduler/nodeinfo:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["predicates_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/apis/core/v1/helper:go_default_library",
        "//pkg/scheduler/framework/plugins/nodename:go_default_library",
        "//pkg/scheduler/framework/plugins/nodeports:go_default_library",
        "//pkg/scheduler/framework/plugins/noderesources:go_default_library",
        "//pkg/scheduler/nodeinfo:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
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
