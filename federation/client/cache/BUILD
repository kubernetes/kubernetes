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
    srcs = ["cluster_cache.go"],
    tags = ["automanaged"],
    deps = [
        "//federation/apis/federation/v1beta1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
