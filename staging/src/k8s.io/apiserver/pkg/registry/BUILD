load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["doc.go"],
    importpath = "k8s.io/apiserver/pkg/registry",
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
    srcs = [
        ":package-srcs",
        "//staging/src/k8s.io/apiserver/pkg/registry/generic:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/registry/rest:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
