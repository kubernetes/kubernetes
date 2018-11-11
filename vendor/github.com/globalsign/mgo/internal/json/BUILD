load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "decode.go",
        "encode.go",
        "extension.go",
        "fold.go",
        "indent.go",
        "scanner.go",
        "stream.go",
        "tags.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/globalsign/mgo/internal/json",
    importpath = "github.com/globalsign/mgo/internal/json",
    visibility = ["//vendor/github.com/globalsign/mgo:__subpackages__"],
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
