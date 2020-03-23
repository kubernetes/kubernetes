load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "pattern.go",
        "readerfactory.go",
        "trie.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/grpc-ecosystem/grpc-gateway/utilities",
    importpath = "github.com/grpc-ecosystem/grpc-gateway/utilities",
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
