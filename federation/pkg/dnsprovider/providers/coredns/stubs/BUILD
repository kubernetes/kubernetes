package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["corednsapi.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/etcd/client",
        "//vendor:golang.org/x/net/context",
    ],
)
