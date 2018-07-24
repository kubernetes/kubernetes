load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "ct.go",
        "ct_ansi.go",
        "ct_win.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/daviddengcn/go-colortext",
    importpath = "github.com/daviddengcn/go-colortext",
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
