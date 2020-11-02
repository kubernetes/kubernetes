load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["peer.go"],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/grpc/peer",
    importpath = "google.golang.org/grpc/peer",
    visibility = ["//visibility:public"],
    deps = ["//vendor/google.golang.org/grpc/credentials:go_default_library"],
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
