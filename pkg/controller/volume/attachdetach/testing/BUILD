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
    srcs = ["testvolumespec.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/fake:go_default_library",
        "//pkg/client/testing/core:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/volume:go_default_library",
        "//pkg/watch:go_default_library",
    ],
)
