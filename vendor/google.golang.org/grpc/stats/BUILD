load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "handlers.go",
        "stats.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/grpc/stats",
    importpath = "google.golang.org/grpc/stats",
    visibility = ["//visibility:public"],
    deps = ["//vendor/google.golang.org/grpc/metadata:go_default_library"],
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
