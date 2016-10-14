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
    srcs = ["registry.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/controller/informers:go_default_library",
        "//pkg/quota:go_default_library",
        "//pkg/quota/evaluator/core:go_default_library",
    ],
)
