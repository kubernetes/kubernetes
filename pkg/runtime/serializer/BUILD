package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["doc.go"],
    tags = ["automanaged"],
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
        "//pkg/runtime/serializer/json:all-srcs",
        "//pkg/runtime/serializer/protobuf:all-srcs",
        "//pkg/runtime/serializer/recognizer:all-srcs",
        "//pkg/runtime/serializer/streaming:all-srcs",
        "//pkg/runtime/serializer/versioning:all-srcs",
    ],
    tags = ["automanaged"],
)
