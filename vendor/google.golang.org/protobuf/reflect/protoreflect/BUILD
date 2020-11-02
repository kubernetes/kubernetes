load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "methods.go",
        "proto.go",
        "source.go",
        "type.go",
        "value.go",
        "value_union.go",
        "value_unsafe.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/protobuf/reflect/protoreflect",
    importpath = "google.golang.org/protobuf/reflect/protoreflect",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/google.golang.org/protobuf/encoding/protowire:go_default_library",
        "//vendor/google.golang.org/protobuf/internal/pragma:go_default_library",
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
