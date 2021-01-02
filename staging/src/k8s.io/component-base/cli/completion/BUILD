package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["completion_test.go"],
    embed = [":go_default_library"],
    deps = ["//vendor/github.com/spf13/cobra:go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = ["completion.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/component-base/cli/completion",
    importpath = "k8s.io/component-base/cli/completion",
    deps = [
        "//vendor/github.com/spf13/cobra:go_default_library",
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
)
