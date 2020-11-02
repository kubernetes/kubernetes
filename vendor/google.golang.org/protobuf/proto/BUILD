load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "checkinit.go",
        "decode.go",
        "decode_gen.go",
        "doc.go",
        "encode.go",
        "encode_gen.go",
        "equal.go",
        "extension.go",
        "merge.go",
        "messageset.go",
        "proto.go",
        "proto_methods.go",
        "reset.go",
        "size.go",
        "size_gen.go",
        "wrappers.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/protobuf/proto",
    importpath = "google.golang.org/protobuf/proto",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/google.golang.org/protobuf/encoding/protowire:go_default_library",
        "//vendor/google.golang.org/protobuf/internal/encoding/messageset:go_default_library",
        "//vendor/google.golang.org/protobuf/internal/errors:go_default_library",
        "//vendor/google.golang.org/protobuf/internal/fieldsort:go_default_library",
        "//vendor/google.golang.org/protobuf/internal/flags:go_default_library",
        "//vendor/google.golang.org/protobuf/internal/mapsort:go_default_library",
        "//vendor/google.golang.org/protobuf/internal/pragma:go_default_library",
        "//vendor/google.golang.org/protobuf/internal/strs:go_default_library",
        "//vendor/google.golang.org/protobuf/reflect/protoreflect:go_default_library",
        "//vendor/google.golang.org/protobuf/reflect/protoregistry:go_default_library",
        "//vendor/google.golang.org/protobuf/runtime/protoiface:go_default_library",
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
