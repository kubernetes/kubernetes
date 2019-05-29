load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["resolver.go"],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/grpc/resolver",
    importpath = "google.golang.org/grpc/resolver",
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
        "//vendor/google.golang.org/grpc/resolver/dns:all-srcs",
        "//vendor/google.golang.org/grpc/resolver/passthrough:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
