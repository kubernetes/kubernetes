load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "blkio_device.go",
        "cgroup_unsupported.go",
        "config.go",
        "device.go",
        "hugepage_limit.go",
        "interface_priority_map.go",
        "mount.go",
        "namespaces.go",
        "namespaces_syscall_unsupported.go",
        "namespaces_unsupported.go",
        "network.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "cgroup_linux.go",
            "config_linux.go",
            "device_defaults.go",
            "namespaces_linux.go",
            "namespaces_syscall.go",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "cgroup_windows.go",
        ],
        "//conditions:default": [],
    }),
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/opencontainers/runtime-spec/specs-go:go_default_library",
        "//vendor/github.com/sirupsen/logrus:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "//conditions:default": [],
    }),
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
    visibility = ["//visibility:public"],
)
