load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "buffer.go",
        "buffer_nopool.go",
        "buffer_pool.go",
        "bytenum.go",
        "decimal.go",
        "extfloat.go",
        "fold.go",
        "ftoa.go",
        "iota.go",
        "jsonstring.go",
        "lexer.go",
        "reader.go",
        "reader_scan_generic.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/pquerna/ffjson/fflib/v1",
    importpath = "github.com/pquerna/ffjson/fflib/v1",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/pquerna/ffjson/fflib/v1/internal:go_default_library"],
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
        "//vendor/github.com/pquerna/ffjson/fflib/v1/internal:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
