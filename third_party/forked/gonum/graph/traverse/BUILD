package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "traverse.go",
        "visit_depth_first.go",
    ],
    importpath = "k8s.io/kubernetes/third_party/forked/gonum/graph/traverse",
    deps = [
        "//third_party/forked/gonum/graph:go_default_library",
        "//third_party/forked/gonum/graph/internal/linear:go_default_library",
        "//vendor/golang.org/x/tools/container/intsets:go_default_library",
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
)
