load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "defs.go",
        "doc.go",
        "helpers.go",
        "http_helpers.go",
        "is.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/errdefs",
    importpath = "github.com/docker/docker/errdefs",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/containerd/containerd/errdefs:go_default_library",
        "//vendor/github.com/docker/distribution/registry/api/errcode:go_default_library",
        "//vendor/github.com/sirupsen/logrus:go_default_library",
        "//vendor/google.golang.org/grpc/codes:go_default_library",
        "//vendor/google.golang.org/grpc/status:go_default_library",
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
