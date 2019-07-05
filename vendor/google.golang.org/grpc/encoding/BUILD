load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["encoding.go"],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/grpc/encoding",
    importpath = "google.golang.org/grpc/encoding",
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
        "//vendor/google.golang.org/grpc/encoding/proto:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
