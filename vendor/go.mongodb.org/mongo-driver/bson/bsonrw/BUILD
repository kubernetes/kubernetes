load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "copier.go",
        "doc.go",
        "extjson_parser.go",
        "extjson_reader.go",
        "extjson_tables.go",
        "extjson_wrappers.go",
        "extjson_writer.go",
        "json_scanner.go",
        "mode.go",
        "reader.go",
        "value_reader.go",
        "value_writer.go",
        "writer.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.mongodb.org/mongo-driver/bson/bsonrw",
    importpath = "go.mongodb.org/mongo-driver/bson/bsonrw",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.mongodb.org/mongo-driver/bson/bsontype:go_default_library",
        "//vendor/go.mongodb.org/mongo-driver/bson/primitive:go_default_library",
        "//vendor/go.mongodb.org/mongo-driver/x/bsonx/bsoncore:go_default_library",
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
