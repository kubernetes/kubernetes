load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "block.go",
        "doc.go",
        "esc.go",
        "html.go",
        "inline.go",
        "markdown.go",
        "node.go",
        "smartypants.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/russross/blackfriday/v2",
    importpath = "github.com/russross/blackfriday/v2",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/shurcooL/sanitized_anchor_name:go_default_library"],
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
