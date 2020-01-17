load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["bsontype.go"],
    importmap = "k8s.io/kubernetes/vendor/go.mongodb.org/mongo-driver/bson/bsontype",
    importpath = "go.mongodb.org/mongo-driver/bson/bsontype",
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
