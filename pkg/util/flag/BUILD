package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["flags.go"],
    importpath = "k8s.io/kubernetes/pkg/util/flag",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/net:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/utils/net:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = ["flags_test.go"],
    embed = [":go_default_library"],
    deps = ["//vendor/github.com/spf13/pflag:go_default_library"],
)
