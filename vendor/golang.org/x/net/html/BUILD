load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "const.go",
        "doc.go",
        "doctype.go",
        "entity.go",
        "escape.go",
        "foreign.go",
        "node.go",
        "parse.go",
        "render.go",
        "token.go",
    ],
    visibility = ["//visibility:public"],
    deps = ["//vendor/golang.org/x/net/html/atom:go_default_library"],
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
        "//vendor/golang.org/x/net/html/atom:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
