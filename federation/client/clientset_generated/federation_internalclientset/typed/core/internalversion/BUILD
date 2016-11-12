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
        "configmap.go",
        "core_client.go",
        "doc.go",
        "event.go",
        "generated_expansion.go",
        "namespace.go",
        "namespace_expansion.go",
        "secret.go",
        "service.go",
    ],
    tags = ["automanaged"],
)
