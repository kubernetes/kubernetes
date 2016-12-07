package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_binary(
    name = "kubernetes-discovery",
    srcs = ["main.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/kubernetes-discovery/pkg/apis/apiregistration/install:go_default_library",
        "//cmd/kubernetes-discovery/pkg/apis/apiregistration/validation:go_default_library",
        "//cmd/kubernetes-discovery/pkg/client/clientset_generated/internalclientset:go_default_library",
        "//cmd/kubernetes-discovery/pkg/client/informers/apiregistration/internalversion:go_default_library",
        "//cmd/kubernetes-discovery/pkg/client/informers/apiregistration/v1alpha1:go_default_library",
        "//cmd/kubernetes-discovery/pkg/client/listers/apiregistration/internalversion:go_default_library",
        "//cmd/kubernetes-discovery/pkg/client/listers/apiregistration/v1alpha1:go_default_library",
        "//cmd/kubernetes-discovery/pkg/cmd/server:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/util/logs:go_default_library",
    ],
)
