load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bson.go",
        "decimal.go",
        "decode.go",
        "encode.go",
        "json.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gopkg.in/mgo.v2/bson",
    importpath = "gopkg.in/mgo.v2/bson",
    visibility = ["//visibility:public"],
    deps = ["//vendor/gopkg.in/mgo.v2/internal/json:go_default_library"],
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
