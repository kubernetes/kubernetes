load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["registry.go"],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/protobuf/reflect/protoregistry",
    importpath = "google.golang.org/protobuf/reflect/protoregistry",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/google.golang.org/protobuf/internal/errors:go_default_library",
        "//vendor/google.golang.org/protobuf/reflect/protoreflect:go_default_library",
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
