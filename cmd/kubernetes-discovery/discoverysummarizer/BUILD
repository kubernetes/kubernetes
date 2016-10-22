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
        "//cmd/kubernetes-discovery/discoverysummarizer/apis/config/v1alpha1:go_default_library",
        "//pkg/api/unversioned:go_default_library",
    ],
)
