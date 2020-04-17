load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "cpu.go",
        "cpuset.go",
        "defaultpath.go",
        "devices.go",
        "freezer.go",
        "fs2.go",
        "io.go",
        "memory.go",
        "pids.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/opencontainers/runc/libcontainer/cgroups/fs2",
    importpath = "github.com/opencontainers/runc/libcontainer/cgroups/fs2",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
        "//vendor/github.com/opencontainers/runc/libcontainer/utils:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
    ] + select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/github.com/cyphar/filepath-securejoin:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/ebpf:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/ebpf/devicefilter:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/fscommon:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/cyphar/filepath-securejoin:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/ebpf:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/ebpf/devicefilter:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/fscommon:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
