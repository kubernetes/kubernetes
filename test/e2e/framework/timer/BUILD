load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["timer.go"],
    importpath = "k8s.io/kubernetes/test/e2e/framework/timer",
    visibility = ["//visibility:public"],
    deps = [
        "//test/e2e/framework:go_default_library",
        "//test/e2e/perftype:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["timer_test.go"],
    embed = [":go_default_library"],
    deps = ["//vendor/github.com/onsi/gomega:go_default_library"],
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
