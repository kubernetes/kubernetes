package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "decoder.go",
        "encoder.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/serializer/streaming:go_default_library",
        "//pkg/watch:go_default_library",
    ],
)

go_test(
    name = "go_default_xtest",
    srcs = [
        "decoder_test.go",
        "encoder_test.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/testapi:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/client/restclient/watch:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/serializer/streaming:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//pkg/watch:go_default_library",
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
)
