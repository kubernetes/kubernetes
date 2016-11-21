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
    srcs = ["repair.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/core/internalversion:go_default_library",
        "//pkg/client/retry:go_default_library",
        "//pkg/registry/core/rangeallocation:go_default_library",
        "//pkg/registry/core/service:go_default_library",
        "//pkg/registry/core/service/portallocator:go_default_library",
        "//pkg/util/net:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/util/wait:go_default_library",
    ],
)
