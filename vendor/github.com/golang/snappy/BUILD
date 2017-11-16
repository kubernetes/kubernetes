load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "decode.go",
        "decode_other.go",
        "encode.go",
        "encode_other.go",
        "snappy.go",
    ] + select({
        "@io_bazel_rules_go//go/platform:darwin_amd64": [
            "decode_amd64.go",
            "decode_amd64.s",
            "encode_amd64.go",
            "encode_amd64.s",
        ],
        "@io_bazel_rules_go//go/platform:linux_amd64": [
            "decode_amd64.go",
            "decode_amd64.s",
            "encode_amd64.go",
            "encode_amd64.s",
        ],
        "@io_bazel_rules_go//go/platform:windows_amd64": [
            "decode_amd64.go",
            "decode_amd64.s",
            "encode_amd64.go",
            "encode_amd64.s",
        ],
        "//conditions:default": [],
    }),
    importpath = "github.com/golang/snappy",
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
