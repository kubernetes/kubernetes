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
    srcs = ["apiserver.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/genericapiserver:go_default_library",
        "//pkg/registry/generic:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/version:go_default_library",
    ],
)
