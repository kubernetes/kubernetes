load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "decode.go",
        "doc.go",
        "encode.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gonum.org/v1/gonum/graph/encoding/dot",
    importpath = "gonum.org/v1/gonum/graph/encoding/dot",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gonum.org/v1/gonum/graph:go_default_library",
        "//vendor/gonum.org/v1/gonum/graph/encoding:go_default_library",
        "//vendor/gonum.org/v1/gonum/graph/formats/dot:go_default_library",
        "//vendor/gonum.org/v1/gonum/graph/formats/dot/ast:go_default_library",
        "//vendor/gonum.org/v1/gonum/graph/internal/ordered:go_default_library",
        "//vendor/gonum.org/v1/gonum/graph/internal/set:go_default_library",
    ],
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
