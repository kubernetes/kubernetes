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
    srcs = ["provider.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/coreos/go-oidc/jose",
        "//vendor:github.com/coreos/go-oidc/key",
        "//vendor:github.com/coreos/go-oidc/oidc",
    ],
)
