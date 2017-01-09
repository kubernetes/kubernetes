package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "options.go",
        "root.go",
        "start.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/capabilities:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//pkg/localkube:go_default_library",
        "//pkg/version:go_default_library",
        "//vendor:github.com/coreos/pkg/capnslog",
        "//vendor:github.com/spf13/cobra",
        "//vendor:github.com/spf13/pflag",
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
