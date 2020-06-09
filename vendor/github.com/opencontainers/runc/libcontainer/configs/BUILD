load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "blkio_device.go",
        "cgroup_linux.go",
        "cgroup_unsupported.go",
        "config.go",
        "config_linux.go",
        "device.go",
        "device_defaults.go",
        "hugepage_limit.go",
        "intelrdt.go",
        "interface_priority_map.go",
        "mount.go",
        "namespaces.go",
        "namespaces_linux.go",
        "namespaces_syscall.go",
        "namespaces_syscall_unsupported.go",
        "namespaces_unsupported.go",
        "network.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/opencontainers/runc/libcontainer/configs",
    importpath = "github.com/opencontainers/runc/libcontainer/configs",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/opencontainers/runtime-spec/specs-go:go_default_library",
        "//vendor/github.com/sirupsen/logrus:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
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
