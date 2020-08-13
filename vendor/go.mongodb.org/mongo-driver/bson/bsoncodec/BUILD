load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bsoncodec.go",
        "default_value_decoders.go",
        "default_value_encoders.go",
        "doc.go",
        "mode.go",
        "pointer_codec.go",
        "proxy.go",
        "registry.go",
        "struct_codec.go",
        "struct_tag_parser.go",
        "types.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.mongodb.org/mongo-driver/bson/bsoncodec",
    importpath = "go.mongodb.org/mongo-driver/bson/bsoncodec",
    visibility = ["//visibility:public"],
    deps = [
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
