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
        "expansion_generated.go",
        "tokenreview.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/errors:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/authentication:go_default_library",
        "//pkg/apis/authentication/v1beta1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/labels:go_default_library",
    ],
)
