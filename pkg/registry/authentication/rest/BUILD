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
    srcs = ["storage_authentication.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/rest:go_default_library",
        "//pkg/apis/authentication:go_default_library",
        "//pkg/apis/authentication/v1beta1:go_default_library",
        "//pkg/auth/authenticator:go_default_library",
        "//pkg/genericapiserver:go_default_library",
        "//pkg/registry:go_default_library",
        "//pkg/registry/authentication/tokenreview:go_default_library",
    ],
)
