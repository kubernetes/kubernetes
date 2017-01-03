package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

go_binary(
    name = "kubernetes-discovery",
    library = ":go_default_library",
    tags = ["automanaged"],
)

go_library(
    name = "go_default_library",
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

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//cmd/kubernetes-discovery/pkg/apis/apiregistration:all-srcs",
        "//cmd/kubernetes-discovery/pkg/apiserver:all-srcs",
        "//cmd/kubernetes-discovery/pkg/client/clientset_generated/clientset:all-srcs",
        "//cmd/kubernetes-discovery/pkg/client/clientset_generated/internalclientset:all-srcs",
        "//cmd/kubernetes-discovery/pkg/client/informers:all-srcs",
        "//cmd/kubernetes-discovery/pkg/client/listers/apiregistration/internalversion:all-srcs",
        "//cmd/kubernetes-discovery/pkg/client/listers/apiregistration/v1alpha1:all-srcs",
        "//cmd/kubernetes-discovery/pkg/cmd/server:all-srcs",
        "//cmd/kubernetes-discovery/pkg/legacy:all-srcs",
        "//cmd/kubernetes-discovery/pkg/registry/apiservice:all-srcs",
    ],
    tags = ["automanaged"],
)
