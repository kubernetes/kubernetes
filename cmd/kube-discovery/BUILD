package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "kube-discovery",
    library = ":go_default_library",
    tags = ["automanaged"],
)

go_library(
    name = "go_default_library",
    srcs = ["kubediscovery.go"],
    tags = ["automanaged"],
    deps = ["//cmd/kube-discovery/app:go_default_library"],
)
