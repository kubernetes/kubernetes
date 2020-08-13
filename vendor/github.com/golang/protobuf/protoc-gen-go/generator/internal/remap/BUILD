load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["remap.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/golang/protobuf/protoc-gen-go/generator/internal/remap",
    importpath = "github.com/golang/protobuf/protoc-gen-go/generator/internal/remap",
    visibility = ["//vendor/github.com/golang/protobuf/protoc-gen-go/generator:__subpackages__"],
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
