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
        "validation.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/kubeapiserver/options:go_default_library",
        "//vendor:github.com/spf13/pflag",
        "//vendor:k8s.io/apiserver/pkg/server/options",
        "//vendor:k8s.io/apiserver/pkg/storage/storagebackend",
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
