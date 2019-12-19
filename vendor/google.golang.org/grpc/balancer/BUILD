load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["balancer.go"],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/grpc/balancer",
    importpath = "google.golang.org/grpc/balancer",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/google.golang.org/grpc/connectivity:go_default_library",
        "//vendor/google.golang.org/grpc/credentials:go_default_library",
        "//vendor/google.golang.org/grpc/internal:go_default_library",
        "//vendor/google.golang.org/grpc/metadata:go_default_library",
        "//vendor/google.golang.org/grpc/resolver:go_default_library",
        "//vendor/google.golang.org/grpc/serviceconfig:go_default_library",
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
        "//vendor/google.golang.org/grpc/balancer/base:all-srcs",
        "//vendor/google.golang.org/grpc/balancer/roundrobin:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
