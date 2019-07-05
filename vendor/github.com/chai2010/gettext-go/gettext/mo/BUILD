load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "encoder.go",
        "file.go",
        "header.go",
        "message.go",
        "util.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/chai2010/gettext-go/gettext/mo",
    importpath = "github.com/chai2010/gettext-go/gettext/mo",
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
