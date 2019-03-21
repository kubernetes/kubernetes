load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "formula.go",
        "table.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/chai2010/gettext-go/gettext/plural",
    importpath = "github.com/chai2010/gettext-go/gettext/plural",
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
