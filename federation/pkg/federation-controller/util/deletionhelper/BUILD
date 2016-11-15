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
    srcs = ["deletion_helper.go"],
    tags = ["automanaged"],
    deps = [
        "//federation/pkg/federation-controller/util:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/record:go_default_library",
        "//pkg/runtime:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
