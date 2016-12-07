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
        "apiservice.go",
        "interface.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/kubernetes-discovery/pkg/apis/apiregistration/v1alpha1:go_default_library",
        "//cmd/kubernetes-discovery/pkg/client/clientset_generated/release_1_5:go_default_library",
        "//cmd/kubernetes-discovery/pkg/client/informers/internalinterfaces:go_default_library",
        "//cmd/kubernetes-discovery/pkg/client/listers/apiregistration/v1alpha1:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/watch:go_default_library",
    ],
)
