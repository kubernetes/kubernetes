load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bson.go",
        "compatibility.go",
        "decimal.go",
        "decode.go",
        "encode.go",
        "json.go",
        "stream.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/globalsign/mgo/bson",
    importpath = "github.com/globalsign/mgo/bson",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/globalsign/mgo/internal/json:go_default_library"],
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
