load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["populate.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/gogo/protobuf/plugin/populate",
    importpath = "github.com/gogo/protobuf/plugin/populate",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/gogo/protobuf/gogoproto:go_default_library",
        "//vendor/github.com/gogo/protobuf/proto:go_default_library",
        "//vendor/github.com/gogo/protobuf/protoc-gen-gogo/descriptor:go_default_library",
        "//vendor/github.com/gogo/protobuf/protoc-gen-gogo/generator:go_default_library",
        "//vendor/github.com/gogo/protobuf/vanity:go_default_library",
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
