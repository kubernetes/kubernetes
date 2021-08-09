load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "cgroup.go",
        "iter.go",
        "link.go",
        "program.go",
        "syscalls.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/cilium/ebpf/link",
    importpath = "github.com/cilium/ebpf/link",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/cilium/ebpf:go_default_library",
        "//vendor/github.com/cilium/ebpf/asm:go_default_library",
        "//vendor/github.com/cilium/ebpf/internal:go_default_library",
        "//vendor/github.com/cilium/ebpf/internal/unix:go_default_library",
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
    visibility = ["//visibility:public"],
)
