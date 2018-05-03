load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["doc.go"],
    importpath = "google.golang.org/grpc/grpclb/grpc_lb_v1",
    visibility = ["//visibility:public"],
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
        "//vendor/google.golang.org/grpc/grpclb/grpc_lb_v1/messages:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
