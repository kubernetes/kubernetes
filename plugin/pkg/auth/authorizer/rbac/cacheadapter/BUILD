package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["rbac_cache_adapter.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apis/rbac:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/informers/informers_generated/rbac/v1alpha1:go_default_library",
        "//pkg/util/runtime:go_default_library",
    ],
)
