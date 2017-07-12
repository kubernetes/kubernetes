package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["interface.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor/k8s.io/sample-apiserver/pkg/client/informers_generated/internalversion/internalinterfaces:go_default_library",
        "//vendor/k8s.io/sample-apiserver/pkg/client/informers_generated/internalversion/wardle/internalversion:go_default_library",
    ],
)
