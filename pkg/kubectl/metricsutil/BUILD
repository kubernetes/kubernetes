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
        "metrics_client.go",
        "metrics_printer.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/api/validation:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/core/internalversion:go_default_library",
        "//pkg/kubectl:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//vendor:k8s.io/heapster/metrics/apis/metrics/v1alpha1",
    ],
)
