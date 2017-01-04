package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["storage.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/errors:go_default_library",
        "//pkg/apis/authentication:go_default_library",
        "//pkg/auth/authenticator:go_default_library",
        "//pkg/genericapiserver/api/request:go_default_library",
        "//pkg/runtime:go_default_library",
    ],
)
