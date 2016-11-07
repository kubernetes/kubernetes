package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["etcd.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/rbac:go_default_library",
        "//pkg/registry/generic:go_default_library",
        "//pkg/registry/generic/registry:go_default_library",
        "//pkg/registry/rbac/rolebinding:go_default_library",
        "//pkg/runtime:go_default_library",
    ],
)
