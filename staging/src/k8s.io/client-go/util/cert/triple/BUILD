package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["triple.go"],
    tags = ["automanaged"],
    deps = ["//vendor/k8s.io/client-go/util/cert:go_default_library"],
)
