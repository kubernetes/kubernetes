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
    srcs = ["uuid.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/types:go_default_library",
        "//vendor:github.com/pborman/uuid",
    ],
)
