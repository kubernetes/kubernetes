load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bsoncore.go",
        "document.go",
        "document_sequence.go",
        "element.go",
        "tables.go",
        "value.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.mongodb.org/mongo-driver/x/bsonx/bsoncore",
    importpath = "go.mongodb.org/mongo-driver/x/bsonx/bsoncore",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/go-stack/stack:go_default_library",
        "//vendor/go.mongodb.org/mongo-driver/bson/bsontype:go_default_library",
        "//vendor/go.mongodb.org/mongo-driver/bson/primitive:go_default_library",
    ],
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
