load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "blkio.go",
        "cpu.go",
        "cpuacct.go",
        "cpuset.go",
        "devices.go",
        "freezer.go",
        "fs.go",
        "hugetlb.go",
        "kmem.go",
        "memory.go",
        "name.go",
        "net_cls.go",
        "net_prio.go",
        "perf_event.go",
        "pids.go",
        "unsupported.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/opencontainers/runc/libcontainer/cgroups/fs",
    importpath = "github.com/opencontainers/runc/libcontainer/cgroups/fs",
    visibility = ["//visibility:public"],
    deps = select({
        "@io_bazel_rules_go//go/platform:android": [
            "//vendor/github.com/moby/sys/mountinfo:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/devices:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/fscommon:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/system:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/utils:go_default_library",
            "//vendor/github.com/pkg/errors:go_default_library",
            "//vendor/golang.org/x/sys/unix:go_default_library",
        ],
        "@io_bazel_rules_go//go/platform:linux": [
            "//vendor/github.com/moby/sys/mountinfo:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/devices:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/fscommon:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/configs:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/system:go_default_library",
            "//vendor/github.com/opencontainers/runc/libcontainer/utils:go_default_library",
            "//vendor/github.com/pkg/errors:go_default_library",
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
