package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "codec_factory.go",
        "negotiated_codec.go",
        "protobuf_extension.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/unversioned:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/serializer/json:go_default_library",
        "//pkg/runtime/serializer/protobuf:go_default_library",
        "//pkg/runtime/serializer/recognizer:go_default_library",
        "//pkg/runtime/serializer/versioning:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["codec_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/unversioned:go_default_library",
        "//pkg/conversion:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/diff:go_default_library",
        "//vendor:github.com/ghodss/yaml",
        "//vendor:github.com/google/gofuzz",
        "//vendor:github.com/spf13/pflag",
    ],
)
