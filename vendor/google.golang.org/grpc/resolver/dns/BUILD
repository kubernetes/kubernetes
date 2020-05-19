load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["dns_resolver.go"],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/grpc/resolver/dns",
    importpath = "google.golang.org/grpc/resolver/dns",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/google.golang.org/grpc/internal/resolver/dns:go_default_library",
        "//vendor/google.golang.org/grpc/resolver:go_default_library",
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
