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
    srcs = ["metrics.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/util/net:go_default_library",
        "//vendor:github.com/emicklei/go-restful",
        "//vendor:github.com/prometheus/client_golang/prometheus",
    ],
)
