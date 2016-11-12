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
        "doc.go",
        "eviction.go",
        "eviction_expansion.go",
        "generated_expansion.go",
        "poddisruptionbudget.go",
        "policy_client.go",
    ],
    tags = ["automanaged"],
)
