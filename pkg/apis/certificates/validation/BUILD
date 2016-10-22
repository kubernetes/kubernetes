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
    srcs = ["validation.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/validation:go_default_library",
        "//pkg/apis/certificates:go_default_library",
        "//pkg/util/cert:go_default_library",
        "//pkg/util/validation/field:go_default_library",
    ],
)
