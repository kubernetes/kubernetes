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
        "decoder.go",
        "encoder.go",
        "generated.pb.go",
        "register.go",
        "types.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/unversioned:go_default_library",
        "//pkg/conversion:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/serializer/streaming:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/gogo/protobuf/proto",
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
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/serializer/streaming:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//pkg/watch:go_default_library",
        "//pkg/watch/versioned:go_default_library",
    ],
)
