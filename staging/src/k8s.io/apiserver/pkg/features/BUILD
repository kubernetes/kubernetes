package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["kube_features.go"],
    tags = ["automanaged"],
    deps = ["//vendor/k8s.io/apiserver/pkg/util/feature:go_default_library"],
)
