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
    srcs = ["prometheus.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/version:go_default_library",
        "//vendor:github.com/prometheus/client_golang/prometheus",
    ],
)
