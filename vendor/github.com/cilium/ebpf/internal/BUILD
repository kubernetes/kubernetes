load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "cpu.go",
        "endian.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/cilium/ebpf/internal",
    importpath = "github.com/cilium/ebpf/internal",
    visibility = ["//vendor/github.com/cilium/ebpf:__subpackages__"],
    deps = ["//vendor/github.com/pkg/errors:go_default_library"],
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
        "//vendor/github.com/cilium/ebpf/internal/unix:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
