load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["exit.go"],
    importmap = "k8s.io/kubernetes/vendor/go.uber.org/zap/internal/exit",
    importpath = "go.uber.org/zap/internal/exit",
    visibility = ["//vendor/go.uber.org/zap:__subpackages__"],
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
