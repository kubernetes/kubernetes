package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "blkio_device.go",
        "cgroup_unix.go",
        "config.go",
        "config_unix.go",
        "device.go",
        "device_defaults.go",
        "hugepage_limit.go",
        "interface_priority_map.go",
        "mount.go",
        "namespaces.go",
        "namespaces_syscall.go",
        "namespaces_unix.go",
        "network.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor/github.com/Sirupsen/logrus:go_default_library"],
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
        "//vendor/github.com/opencontainers/runc/libcontainer/configs/validate:all-srcs",
    ],
    tags = ["automanaged"],
)
