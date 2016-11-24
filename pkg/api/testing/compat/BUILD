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
    srcs = ["compatibility_tester.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/kubectl:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/util/validation/field:go_default_library",
    ],
)
