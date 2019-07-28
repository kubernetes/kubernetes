load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bron_kerbosch.go",
        "clique_graph.go",
        "doc.go",
        "johnson_cycles.go",
        "non_tomita_choice.go",
        "paton_cycles.go",
        "tarjan.go",
        "topo.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gonum.org/v1/gonum/graph/topo",
    importpath = "gonum.org/v1/gonum/graph/topo",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/gonum.org/v1/gonum/graph:go_default_library",
        "//vendor/gonum.org/v1/gonum/graph/internal/linear:go_default_library",
        "//vendor/gonum.org/v1/gonum/graph/internal/ordered:go_default_library",
        "//vendor/gonum.org/v1/gonum/graph/internal/set:go_default_library",
        "//vendor/gonum.org/v1/gonum/graph/iterator:go_default_library",
        "//vendor/gonum.org/v1/gonum/graph/traverse:go_default_library",
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
