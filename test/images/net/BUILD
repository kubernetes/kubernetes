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
    name = "net",
    srcs = ["main.go"],
    tags = ["automanaged"],
    deps = [
        "//test/images/net/common:go_default_library",
        "//test/images/net/nat:go_default_library",
    ],
)
