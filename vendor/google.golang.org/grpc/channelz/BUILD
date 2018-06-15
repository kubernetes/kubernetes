load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "funcs.go",
        "types.go",
    ],
    importpath = "google.golang.org/grpc/channelz",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/google.golang.org/grpc/connectivity:go_default_library",
        "//vendor/google.golang.org/grpc/grpclog:go_default_library",
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
