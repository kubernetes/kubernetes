load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["sanitization.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/component-base/logs/sanitization",
    importpath = "k8s.io/component-base/logs/sanitization",
    visibility = ["//visibility:public"],
    deps = ["//staging/src/k8s.io/component-base/logs/datapol:go_default_library"],
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
    srcs = ["sanitization_test.go"],
    embed = [":go_default_library"],
    deps = ["//vendor/github.com/stretchr/testify/assert:go_default_library"],
)
