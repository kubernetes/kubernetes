load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = select({
        "@io_bazel_rules_go//go/platform:android": [
            "fs_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:darwin": [
            "fs_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:dragonfly": [
            "fs_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:freebsd": [
            "fs_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "apply_raw.go",
            "blkio.go",
            "cpu.go",
            "cpuacct.go",
            "cpuset.go",
            "devices.go",
            "freezer.go",
            "hugetlb.go",
            "memory.go",
            "name.go",
            "net_cls.go",
            "net_prio.go",
            "perf_event.go",
            "pids.go",
            "utils.go",
        ],
        "@io_bazel_rules_go//go/platform:nacl": [
            "fs_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:netbsd": [
            "fs_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:openbsd": [
            "fs_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:plan9": [
            "fs_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:solaris": [
            "fs_unsupported.go",
        ],
        "@io_bazel_rules_go//go/platform:windows": [
            "fs_unsupported.go",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/opencontainers/runc/libcontainer/cgroups/fs",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/system:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/utils:go_default_library",
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
