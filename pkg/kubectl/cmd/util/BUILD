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
        "cached_discovery.go",
        "clientcache.go",
        "factory.go",
        "helpers.go",
        "printing.go",
        "shortcut_restmapper.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "cached_discovery_test.go",
        "factory_test.go",
        "helpers_test.go",
        "shortcut_restmapper_test.go",
    ],
    data = [
        "//api/swagger-spec",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
