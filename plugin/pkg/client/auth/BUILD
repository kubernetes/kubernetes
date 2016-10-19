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
    srcs = ["plugins.go"],
    tags = ["automanaged"],
    deps = [
        "//plugin/pkg/client/auth/gcp:go_default_library",
        "//plugin/pkg/client/auth/oidc:go_default_library",
    ],
)
