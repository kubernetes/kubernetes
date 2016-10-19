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
    srcs = ["helpers.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/authorization:go_default_library",
        "//pkg/auth/authorizer:go_default_library",
        "//pkg/auth/user:go_default_library",
    ],
)
