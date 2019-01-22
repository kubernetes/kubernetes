load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bool.go",
        "doc.go",
        "enum.go",
        "int.go",
        "scan.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gopkg.in/gcfg.v1/types",
    importpath = "gopkg.in/gcfg.v1/types",
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
