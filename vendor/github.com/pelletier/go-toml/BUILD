load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "keysparsing.go",
        "lexer.go",
        "match.go",
        "parser.go",
        "position.go",
        "query.go",
        "querylexer.go",
        "queryparser.go",
        "token.go",
        "toml.go",
        "tomltree_conversions.go",
    ],
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/pelletier/go-buffruneio:go_default_library"],
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
