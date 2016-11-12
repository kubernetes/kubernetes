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
        "fake_configmap.go",
        "fake_core_client.go",
        "fake_event.go",
        "fake_namespace.go",
        "fake_namespace_expansion.go",
        "fake_secret.go",
        "fake_service.go",
    ],
    tags = ["automanaged"],
)
