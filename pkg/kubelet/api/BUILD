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
    srcs = ["services.go"],
    tags = ["automanaged"],
    deps = ["//pkg/kubelet/api/v1alpha1/runtime:go_default_library"],
)
