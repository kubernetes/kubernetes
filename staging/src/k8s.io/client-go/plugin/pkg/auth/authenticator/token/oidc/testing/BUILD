package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["provider.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/coreos/go-oidc/jose:go_default_library",
        "//vendor/github.com/coreos/go-oidc/key:go_default_library",
        "//vendor/github.com/coreos/go-oidc/oidc:go_default_library",
    ],
)
