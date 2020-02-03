load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "decode.go",
        "decode_amd64.go",
        "decode_amd64.s",
        "decode_other.go",
        "encode.go",
        "encode_amd64.go",
        "encode_amd64.s",
        "encode_other.go",
        "snappy.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/golang/snappy",
    importpath = "github.com/golang/snappy",
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
