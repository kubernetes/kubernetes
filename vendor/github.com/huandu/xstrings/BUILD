load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "common.go",
        "convert.go",
        "count.go",
        "doc.go",
        "format.go",
        "manipulate.go",
        "translate.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/huandu/xstrings",
    importpath = "github.com/huandu/xstrings",
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
