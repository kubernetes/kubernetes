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
    name = "controller",
    srcs = ["controller.go"],
    tags = ["automanaged"],
    deps = ["//test/images/resource-consumer/common:go_default_library"],
)
