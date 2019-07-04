load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "credentials.go",
        "credentials_util_go17.go",
        "credentials_util_go18.go",
        "credentials_util_pre_go17.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/google.golang.org/grpc/credentials",
    importpath = "google.golang.org/grpc/credentials",
    visibility = ["//visibility:public"],
    deps = ["//vendor/golang.org/x/net/context:go_default_library"],
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
