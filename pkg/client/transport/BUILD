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
        "cache.go",
        "config.go",
        "round_trippers.go",
        "transport.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/util/net:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "cache_test.go",
        "round_trippers_test.go",
        "transport_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [],
)
