load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "asm.go",
        "constants.go",
        "doc.go",
        "instructions.go",
        "setter.go",
        "vm.go",
        "vm_instructions.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/net/bpf",
    importpath = "golang.org/x/net/bpf",
    visibility = ["//visibility:public"],
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
