load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["azure_error.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/legacy-cloud-providers/azure/retry",
    importpath = "k8s.io/legacy-cloud-providers/azure/retry",
    visibility = ["//visibility:public"],
    deps = ["//vendor/k8s.io/klog:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = ["azure_error_test.go"],
    embed = [":go_default_library"],
    deps = ["//vendor/github.com/stretchr/testify/assert:go_default_library"],
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
