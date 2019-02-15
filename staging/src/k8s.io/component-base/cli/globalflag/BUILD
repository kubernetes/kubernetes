load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["globalflags.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/component-base/cli/globalflag",
    importpath = "k8s.io/component-base/cli/globalflag",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/component-base/logs:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["globalflags_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/component-base/cli/flag:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
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
