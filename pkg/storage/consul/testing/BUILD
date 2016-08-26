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
    srcs = ["utils.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/storage/consul/consultest:go_default_library",
        "//pkg/storage/storagebackend:go_default_library",
        "//vendor:github.com/hashicorp/consul/api",
        "//vendor:github.com/hashicorp/consul/command/agent",
        "//vendor:github.com/hashicorp/consul/consul",
    ],
)
