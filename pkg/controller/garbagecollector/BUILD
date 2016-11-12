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
        "garbagecollector.go",
        "metrics.go",
        "rate_limiter_helper.go",
        "uid_cache.go",
    ],
    tags = ["automanaged"],
)

go_test(
    name = "go_default_test",
    srcs = ["garbagecollector_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
