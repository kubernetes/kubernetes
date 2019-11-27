load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "decode.go",
        "decode_meta.go",
        "doc.go",
        "encode.go",
        "encoding_types.go",
        "encoding_types_1.1.go",
        "lex.go",
        "parse.go",
        "type_check.go",
        "type_fields.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/BurntSushi/toml",
    importpath = "github.com/BurntSushi/toml",
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
