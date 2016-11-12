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
    srcs = [
        "addons.go",
        "apiclient.go",
        "discovery.go",
        "kubeconfig.go",
        "manifests.go",
        "pki.go",
        "tokens.go",
    ],
    tags = ["automanaged"],
)
