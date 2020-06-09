load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "abi.go",
        "collection.go",
        "doc.go",
        "elf_reader.go",
        "feature.go",
        "linker.go",
        "map.go",
        "marshalers.go",
        "prog.go",
        "ptr_32_be.go",
        "ptr_32_le.go",
        "ptr_64.go",
        "syscalls.go",
        "types.go",
        "types_string.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/cilium/ebpf",
    importpath = "github.com/cilium/ebpf",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/cilium/ebpf/asm:go_default_library",
        "//vendor/github.com/cilium/ebpf/internal:go_default_library",
        "//vendor/github.com/cilium/ebpf/internal/unix:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
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
        "//vendor/github.com/cilium/ebpf/asm:all-srcs",
        "//vendor/github.com/cilium/ebpf/internal:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
