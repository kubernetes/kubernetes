load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "capability.go",
        "capability_linux.go",
        "capability_noop.go",
        "enum.go",
        "enum_gen.go",
        "syscall_linux.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/syndtr/gocapability/capability",
    importpath = "github.com/syndtr/gocapability/capability",
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
