load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "decode.go",
        "doc.go",
        "integrate.go",
        "lex.go",
        "load.go",
        "parser.go",
        "properties.go",
        "rangecheck.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/magiconair/properties",
    importpath = "github.com/magiconair/properties",
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
