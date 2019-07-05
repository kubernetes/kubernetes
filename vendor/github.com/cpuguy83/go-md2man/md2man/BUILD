load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "md2man.go",
        "roff.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/cpuguy83/go-md2man/md2man",
    importpath = "github.com/cpuguy83/go-md2man/md2man",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/russross/blackfriday:go_default_library"],
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
