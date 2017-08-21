load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "block.go",
        "html.go",
        "inline.go",
        "latex.go",
        "markdown.go",
        "smartypants.go",
    ],
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
