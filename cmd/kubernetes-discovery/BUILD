package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_binary(
    name = "kubernetes-discovery",
    srcs = ["main.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/kubernetes-discovery/discoverysummarizer:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
