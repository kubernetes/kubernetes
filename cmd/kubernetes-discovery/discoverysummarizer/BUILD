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
        "discoverysummarizer.go",
        "doc.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//apifederator/pkg/apis/apifederation/install:go_default_library",
        "//apifederator/pkg/apis/apifederation/validation:go_default_library",
        "//apifederator/pkg/client/clientset_generated/internalclientset:go_default_library",
        "//apifederator/pkg/client/listers/apifederation/internalversion:go_default_library",
        "//apifederator/pkg/client/listers/apifederation/v1alpha1:go_default_library",
        "//cmd/kubernetes-discovery/discoverysummarizer/apis/config/v1alpha1:go_default_library",
        "//pkg/api/unversioned:go_default_library",
    ],
)
