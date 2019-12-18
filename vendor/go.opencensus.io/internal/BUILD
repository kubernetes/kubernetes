load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "internal.go",
        "sanitize.go",
        "traceinternals.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.opencensus.io/internal",
    importpath = "go.opencensus.io/internal",
    visibility = ["//vendor/go.opencensus.io:__subpackages__"],
    deps = ["//vendor/go.opencensus.io:go_default_library"],
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
        "//vendor/go.opencensus.io/internal/tagencoding:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
