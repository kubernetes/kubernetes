load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["stream_chunk.pb.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/grpc-ecosystem/grpc-gateway/internal",
    importpath = "github.com/grpc-ecosystem/grpc-gateway/internal",
    visibility = ["//vendor/github.com/grpc-ecosystem/grpc-gateway:__subpackages__"],
    deps = [
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/github.com/golang/protobuf/ptypes/any:go_default_library",
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
