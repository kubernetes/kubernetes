load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "progress.go",
        "progressreader.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/pkg/progress",
    importpath = "github.com/docker/docker/pkg/progress",
    visibility = ["//visibility:public"],
    deps = ["//vendor/golang.org/x/time/rate:go_default_library"],
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
