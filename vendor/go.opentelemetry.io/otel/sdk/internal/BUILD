load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "internal.go",
        "sanitize.go",
        "state_locker.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/go.opentelemetry.io/otel/sdk/internal",
    importpath = "go.opentelemetry.io/otel/sdk/internal",
    visibility = ["//vendor/go.opentelemetry.io/otel/sdk:__subpackages__"],
    deps = ["//vendor/go.opentelemetry.io/otel/sdk:go_default_library"],
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
