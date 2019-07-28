load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["grpcrand.go"],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/grpc/internal/grpcrand",
    importpath = "google.golang.org/grpc/internal/grpcrand",
    visibility = ["//vendor/google.golang.org/grpc:__subpackages__"],
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
