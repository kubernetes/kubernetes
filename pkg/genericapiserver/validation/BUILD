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
        "etcd_validation.go",
        "universal_validation.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/genericapiserver/options:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
