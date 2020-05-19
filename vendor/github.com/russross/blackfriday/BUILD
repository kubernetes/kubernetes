load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "block.go",
        "doc.go",
        "html.go",
        "inline.go",
        "latex.go",
        "markdown.go",
        "smartypants.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/russross/blackfriday",
    importpath = "github.com/russross/blackfriday",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/russross/blackfriday/v2:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
