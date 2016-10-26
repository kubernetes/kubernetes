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
        "zz_generated.clusterrole.go",
        "zz_generated.clusterrolebinding.go",
        "zz_generated.role.go",
        "zz_generated.rolebinding.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/apis/rbac:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/labels:go_default_library",
    ],
)
