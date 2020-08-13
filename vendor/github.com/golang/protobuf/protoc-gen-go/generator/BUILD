load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["generator.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/golang/protobuf/protoc-gen-go/generator",
    importpath = "github.com/golang/protobuf/protoc-gen-go/generator",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/golang/protobuf/proto:go_default_library",
        "//vendor/github.com/golang/protobuf/protoc-gen-go/descriptor:go_default_library",
        "//vendor/github.com/golang/protobuf/protoc-gen-go/generator/internal/remap:go_default_library",
        "//vendor/github.com/golang/protobuf/protoc-gen-go/plugin:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/golang/protobuf/protoc-gen-go/generator/internal/remap:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
