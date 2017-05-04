package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "clientset_generated.go",
        "doc.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/client-go/discovery:go_default_library",
        "//vendor/k8s.io/client-go/discovery/fake:go_default_library",
        "//vendor/k8s.io/client-go/pkg/api:go_default_library",
        "//vendor/k8s.io/client-go/testing:go_default_library",
        "//vendor/k8s.io/metrics/pkg/client/clientset_generated/clientset:go_default_library",
        "//vendor/k8s.io/metrics/pkg/client/clientset_generated/clientset/typed/metrics/v1alpha1:go_default_library",
        "//vendor/k8s.io/metrics/pkg/client/clientset_generated/clientset/typed/metrics/v1alpha1/fake:go_default_library",
    ],
)
