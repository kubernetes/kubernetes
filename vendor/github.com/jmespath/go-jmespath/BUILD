load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "api.go",
        "astnodetype_string.go",
        "functions.go",
        "interpreter.go",
        "lexer.go",
        "parser.go",
        "toktype_string.go",
        "util.go",
    ],
    importpath = "github.com/jmespath/go-jmespath",
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
        "//vendor/github.com/jmespath/go-jmespath/cmd/jpgo:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
