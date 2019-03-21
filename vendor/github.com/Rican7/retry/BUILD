load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["retry.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/Rican7/retry",
    importpath = "github.com/Rican7/retry",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/Rican7/retry/strategy:go_default_library"],
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
        "//vendor/github.com/Rican7/retry/backoff:all-srcs",
        "//vendor/github.com/Rican7/retry/jitter:all-srcs",
        "//vendor/github.com/Rican7/retry/strategy:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
