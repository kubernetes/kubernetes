load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bson.go",
        "bson_1_8.go",
        "decoder.go",
        "doc.go",
        "encoder.go",
        "marshal.go",
        "primitive_codecs.go",
        "raw.go",
        "raw_element.go",
        "raw_value.go",
        "registry.go",
        "types.go",
        "unmarshal.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.mongodb.org/mongo-driver/bson",
    importpath = "go.mongodb.org/mongo-driver/bson",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/go.mongodb.org/mongo-driver/bson/bsoncodec:go_default_library",
        "//vendor/go.mongodb.org/mongo-driver/bson/bsonrw:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/go.mongodb.org/mongo-driver/bson/bsoncodec:all-srcs",
        "//vendor/go.mongodb.org/mongo-driver/bson/bsonrw:all-srcs",
        "//vendor/go.mongodb.org/mongo-driver/bson/bsontype:all-srcs",
        "//vendor/go.mongodb.org/mongo-driver/bson/primitive:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
