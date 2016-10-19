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
    srcs = ["plugins.go"],
    tags = ["automanaged"],
    deps = ["//plugin/pkg/scheduler/algorithmprovider/defaults:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = ["plugins_test.go"],
    library = "go_default_library",
    tags = [
        "automanaged",
        "skip",
    ],
    deps = ["//plugin/pkg/scheduler/factory:go_default_library"],
)
