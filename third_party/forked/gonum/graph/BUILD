package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["graph.go"],
    importpath = "k8s.io/kubernetes/third_party/forked/gonum/graph",
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
        "//third_party/forked/gonum/graph/internal/linear:all-srcs",
        "//third_party/forked/gonum/graph/simple:all-srcs",
        "//third_party/forked/gonum/graph/traverse:all-srcs",
    ],
    tags = ["automanaged"],
)
