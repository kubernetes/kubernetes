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
    srcs = ["interface.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/kubernetes-discovery/pkg/client/informers/apiregistration/internalversion:go_default_library",
        "//cmd/kubernetes-discovery/pkg/client/informers/apiregistration/v1alpha1:go_default_library",
        "//cmd/kubernetes-discovery/pkg/client/informers/internalinterfaces:go_default_library",
    ],
)
