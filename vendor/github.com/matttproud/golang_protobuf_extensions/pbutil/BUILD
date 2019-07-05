load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "decode.go",
        "doc.go",
        "encode.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/matttproud/golang_protobuf_extensions/pbutil",
    importpath = "github.com/matttproud/golang_protobuf_extensions/pbutil",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/golang/protobuf/proto:go_default_library"],
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
