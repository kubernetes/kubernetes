load("@io_bazel_rules_go//go:def.bzl", "go_library")

filegroup(
    name = "go_default_library_protos",
    srcs = ["messages.proto"],
    visibility = ["//visibility:public"],
)

go_library(
    name = "go_default_library",
    srcs = ["messages.pb.go"],
    importpath = "google.golang.org/grpc/grpclb/grpc_lb_v1/messages",
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
