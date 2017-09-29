load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "clone.go",
        "decode.go",
        "decode_gogo.go",
        "duration.go",
        "duration_gogo.go",
        "encode.go",
        "encode_gogo.go",
        "equal.go",
        "extensions.go",
        "extensions_gogo.go",
        "lib.go",
        "lib_gogo.go",
        "message_set.go",
        "pointer_unsafe.go",
        "pointer_unsafe_gogo.go",
        "properties.go",
        "properties_gogo.go",
        "skip_gogo.go",
        "text.go",
        "text_gogo.go",
        "text_parser.go",
        "timestamp.go",
        "timestamp_gogo.go",
    ],
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
