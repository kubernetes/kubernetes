package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "counter.go",
        "doc.go",
        "string.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = ["maps_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
