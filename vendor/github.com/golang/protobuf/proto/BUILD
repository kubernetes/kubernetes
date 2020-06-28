load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "buffer.go",
        "defaults.go",
        "deprecated.go",
        "discard.go",
        "extensions.go",
        "properties.go",
        "proto.go",
        "registry.go",
        "text_decode.go",
        "text_encode.go",
        "wire.go",
        "wrappers.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/golang/protobuf/proto",
    importpath = "github.com/golang/protobuf/proto",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/google.golang.org/protobuf/encoding/prototext:go_default_library",
        "//vendor/google.golang.org/protobuf/encoding/protowire:go_default_library",
        "//vendor/google.golang.org/protobuf/proto:go_default_library",
        "//vendor/google.golang.org/protobuf/reflect/protoreflect:go_default_library",
        "//vendor/google.golang.org/protobuf/reflect/protoregistry:go_default_library",
        "//vendor/google.golang.org/protobuf/runtime/protoiface:go_default_library",
        "//vendor/google.golang.org/protobuf/runtime/protoimpl:go_default_library",
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
