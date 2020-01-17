load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["syscallconn.go"],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/grpc/credentials/internal",
    importpath = "google.golang.org/grpc/credentials/internal",
    visibility = ["//vendor/google.golang.org/grpc/credentials:__subpackages__"],
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
