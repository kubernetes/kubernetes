load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["main.go"],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/structured-merge-diff/smd",
    importpath = "sigs.k8s.io/structured-merge-diff/smd",
    visibility = ["//visibility:private"],
    deps = ["//vendor/sigs.k8s.io/structured-merge-diff/internal/cli:go_default_library"],
)

go_binary(
    name = "smd",
    embed = [":go_default_library"],
    tags = ["manual"],
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
