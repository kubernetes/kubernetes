load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "clone.go",
        "decode.go",
        "discard.go",
        "encode.go",
        "equal.go",
        "extensions.go",
        "lib.go",
        "message_set.go",
        "pointer_unsafe.go",
        "properties.go",
        "table_marshal.go",
        "table_merge.go",
        "table_unmarshal.go",
        "text.go",
        "text_parser.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/golang/protobuf/proto",
    importpath = "github.com/golang/protobuf/proto",
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
