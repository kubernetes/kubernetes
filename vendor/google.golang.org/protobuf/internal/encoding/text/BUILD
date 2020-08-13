load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "decode.go",
        "decode_number.go",
        "decode_string.go",
        "decode_token.go",
        "doc.go",
        "encode.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/protobuf/internal/encoding/text",
    importpath = "google.golang.org/protobuf/internal/encoding/text",
    visibility = ["//vendor/google.golang.org/protobuf:__subpackages__"],
    deps = [
        "//vendor/google.golang.org/protobuf/internal/detrand:go_default_library",
        "//vendor/google.golang.org/protobuf/internal/errors:go_default_library",
        "//vendor/google.golang.org/protobuf/internal/flags:go_default_library",
        "//vendor/google.golang.org/protobuf/internal/strs:go_default_library",
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
