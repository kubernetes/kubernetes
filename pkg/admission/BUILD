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
        "attributes.go",
        "chain.go",
        "errors.go",
        "handler.go",
        "init.go",
        "interfaces.go",
        "plugins.go",
        "types.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "chain_test.go",
        "init_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
