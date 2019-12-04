load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["utils.go"],
    importpath = "k8s.io/kubernetes/pkg/scheduler/framework/plugins/migration",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/scheduler/algorithm/predicates:go_default_library",
        "//pkg/scheduler/framework/v1alpha1:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = ["utils_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/scheduler/algorithm/predicates:go_default_library",
        "//pkg/scheduler/framework/v1alpha1:go_default_library",
    ],
)
