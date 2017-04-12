package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "cgroups.go",
        "stats.go",
        "utils.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/docker/go-units:go_default_library",
        "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/fs:all-srcs",
        "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/systemd:all-srcs",
    ],
    tags = ["automanaged"],
)
