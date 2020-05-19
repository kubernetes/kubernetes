load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["plugin.pb.go"],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/protobuf/types/pluginpb",
    importpath = "google.golang.org/protobuf/types/pluginpb",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/google.golang.org/protobuf/reflect/protoreflect:go_default_library",
        "//vendor/google.golang.org/protobuf/runtime/protoimpl:go_default_library",
        "//vendor/google.golang.org/protobuf/types/descriptorpb:go_default_library",
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
