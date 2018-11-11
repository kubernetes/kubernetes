load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "enum.go",
        "field.go",
        "file.go",
        "foreach.go",
        "msg.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/gogo/protobuf/vanity",
    importpath = "github.com/gogo/protobuf/vanity",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/gogo/protobuf/gogoproto:go_default_library",
        "//vendor/github.com/gogo/protobuf/proto:go_default_library",
        "//vendor/github.com/gogo/protobuf/protoc-gen-gogo/descriptor:go_default_library",
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
        "//vendor/github.com/gogo/protobuf/vanity/command:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
