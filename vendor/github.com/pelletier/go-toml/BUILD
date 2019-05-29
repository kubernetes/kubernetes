load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "keysparsing.go",
        "lexer.go",
        "marshal.go",
        "parser.go",
        "position.go",
        "token.go",
        "toml.go",
        "tomltree_create.go",
        "tomltree_write.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/pelletier/go-toml",
    importpath = "github.com/pelletier/go-toml",
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
