package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "fs_resource_analyzer.go",
        "handler.go",
        "resource_analyzer.go",
        "summary.go",
        "volume_stat_calculator.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "mocks_test.go",
        "summary_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
