package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["discoverysummarizer_test.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/kubernetes-discovery/pkg/legacy:go_default_library",
        "//examples/apiserver:go_default_library",
    ],
)
