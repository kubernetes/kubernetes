load("@io_bazel_rules_go//go:def.bzl", "go_library")

licenses(["notice"])

go_library(
    name = "go_default_library",
    srcs = ["srcimporter.go"],
    importpath = "k8s.io/kubernetes/third_party/go-srcimporter",
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
