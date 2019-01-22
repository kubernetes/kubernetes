load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "token.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gonum.org/v1/gonum/graph/formats/dot/internal/token",
    importpath = "gonum.org/v1/gonum/graph/formats/dot/internal/token",
    visibility = ["//vendor/gonum.org/v1/gonum/graph/formats/dot:__subpackages__"],
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
