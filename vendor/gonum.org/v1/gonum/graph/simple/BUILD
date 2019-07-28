load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "dense_directed_matrix.go",
        "dense_undirected_matrix.go",
        "directed.go",
        "doc.go",
        "simple.go",
        "undirected.go",
        "weighted_directed.go",
        "weighted_undirected.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gonum.org/v1/gonum/graph/simple",
    importpath = "gonum.org/v1/gonum/graph/simple",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gonum.org/v1/gonum/graph:go_default_library",
        "//vendor/gonum.org/v1/gonum/graph/internal/ordered:go_default_library",
        "//vendor/gonum.org/v1/gonum/graph/internal/uid:go_default_library",
        "//vendor/gonum.org/v1/gonum/graph/iterator:go_default_library",
        "//vendor/gonum.org/v1/gonum/mat:go_default_library",
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
