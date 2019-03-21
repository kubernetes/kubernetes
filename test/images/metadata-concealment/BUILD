load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["check_metadata_concealment.go"],
    importpath = "k8s.io/kubernetes/test/images/metadata-concealment",
    visibility = ["//visibility:private"],
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

go_binary(
    name = "metadata-concealment",
    embed = [":go_default_library"],
    visibility = ["//visibility:public"],
)
