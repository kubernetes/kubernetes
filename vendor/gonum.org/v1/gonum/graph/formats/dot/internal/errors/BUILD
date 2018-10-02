load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "errors.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gonum.org/v1/gonum/graph/formats/dot/internal/errors",
    importpath = "gonum.org/v1/gonum/graph/formats/dot/internal/errors",
    visibility = ["//vendor/gonum.org/v1/gonum/graph/formats/dot:__subpackages__"],
    deps = ["//vendor/gonum.org/v1/gonum/graph/formats/dot/internal/token:go_default_library"],
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
