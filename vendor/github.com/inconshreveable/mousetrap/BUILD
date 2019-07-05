load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "trap_others.go",
        "trap_windows.go",
        "trap_windows_1.4.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/inconshreveable/mousetrap",
    importpath = "github.com/inconshreveable/mousetrap",
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
