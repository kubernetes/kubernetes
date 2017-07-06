package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "directed_acyclic_test.go",
        "edgeholder_test.go",
        "undirected_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//third_party/forked/gonum/graph:go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "directed_acyclic.go",
        "edgeholder.go",
        "simple.go",
        "undirected.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//third_party/forked/gonum/graph:go_default_library",
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
