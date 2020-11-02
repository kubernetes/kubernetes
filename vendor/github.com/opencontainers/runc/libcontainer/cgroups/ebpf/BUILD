load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["ebpf.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/opencontainers/runc/libcontainer/cgroups/ebpf",
    importpath = "github.com/opencontainers/runc/libcontainer/cgroups/ebpf",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/cilium/ebpf:go_default_library",
        "//vendor/github.com/cilium/ebpf/asm:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/golang.org/x/sys/unix:go_default_library",
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
        "//vendor/github.com/opencontainers/runc/libcontainer/cgroups/ebpf/devicefilter:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
