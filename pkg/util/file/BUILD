package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["file.go"],
    importpath = "k8s.io/kubernetes/pkg/util/file",
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
    srcs = ["file_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//vendor/github.com/spf13/afero:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
    ],
)
