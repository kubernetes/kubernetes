package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "config.go",
        "host_config.go",
        "hostconfig_unix.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/docker/engine-api/types/blkiodev:go_default_library",
        "//vendor/github.com/docker/engine-api/types/strslice:go_default_library",
        "//vendor/github.com/docker/go-connections/nat:go_default_library",
        "//vendor/github.com/docker/go-units:go_default_library",
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
