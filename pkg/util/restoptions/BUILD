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
    srcs = ["helpers.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/genericapiserver:go_default_library",
        "//pkg/registry/cachesize:go_default_library",
        "//pkg/registry/generic/registry:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/storage:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
