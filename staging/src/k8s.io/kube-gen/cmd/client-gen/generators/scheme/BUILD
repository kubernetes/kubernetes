package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["generator_for_scheme.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor/k8s.io/gengo/generator:go_default_library",
        "//vendor/k8s.io/gengo/namer:go_default_library",
        "//vendor/k8s.io/gengo/types:go_default_library",
        "//vendor/k8s.io/kube-gen/cmd/client-gen/path:go_default_library",
        "//vendor/k8s.io/kube-gen/cmd/client-gen/types:go_default_library",
    ],
)
