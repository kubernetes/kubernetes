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
        "clusterrole.go",
        "clusterrolebinding.go",
        "expansion_generated.go",
        "role.go",
        "rolebinding.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/errors:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/rbac:go_default_library",
        "//pkg/apis/rbac/v1alpha1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/labels:go_default_library",
    ],
)
