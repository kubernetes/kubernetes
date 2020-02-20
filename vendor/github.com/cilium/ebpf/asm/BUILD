load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "alu.go",
        "alu_string.go",
        "doc.go",
        "func.go",
        "func_string.go",
        "instruction.go",
        "jump.go",
        "jump_string.go",
        "load_store.go",
        "load_store_string.go",
        "opcode.go",
        "opcode_string.go",
        "register.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/cilium/ebpf/asm",
    importpath = "github.com/cilium/ebpf/asm",
    visibility = ["//visibility:public"],
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
