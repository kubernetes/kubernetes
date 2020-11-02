load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "encode.go",
        "hpack.go",
        "huffman.go",
        "tables.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/net/http2/hpack",
    importpath = "golang.org/x/net/http2/hpack",
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
