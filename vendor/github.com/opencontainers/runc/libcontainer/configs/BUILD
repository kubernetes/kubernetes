load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "blkio_device.go",
        "config.go",
        "device.go",
        "hugepage_limit.go",
        "intelrdt.go",
        "interface_priority_map.go",
        "mount.go",
        "namespaces.go",
        "network.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "namespaces_syscall_unsupported.go",
            "namespaces_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "namespaces_syscall_unsupported.go",
            "namespaces_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "namespaces_syscall_unsupported.go",
            "namespaces_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "namespaces_syscall_unsupported.go",
            "namespaces_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "cgroup_linux.go",
            "config_linux.go",
            "device_defaults.go",
            "namespaces_linux.go",
            "namespaces_syscall.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "namespaces_syscall_unsupported.go",
            "namespaces_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "namespaces_syscall_unsupported.go",
            "namespaces_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "namespaces_syscall_unsupported.go",
            "namespaces_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "namespaces_syscall_unsupported.go",
            "namespaces_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "namespaces_syscall_unsupported.go",
            "namespaces_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "cgroup_windows.go",
            "namespaces_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/opencontainers/runc/libcontainer/configs",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/opencontainers/runtime-spec/specs-go:go_default_library",
        "//vendor/github.com/sirupsen/logrus:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:linux": [
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
