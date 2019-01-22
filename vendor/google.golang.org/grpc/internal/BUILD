load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["internal.go"],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/grpc/internal",
    importpath = "google.golang.org/grpc/internal",
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
        "//vendor/google.golang.org/grpc/internal/backoff:all-srcs",
        "//vendor/google.golang.org/grpc/internal/channelz:all-srcs",
        "//vendor/google.golang.org/grpc/internal/grpcrand:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
