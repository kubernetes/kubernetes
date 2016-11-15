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
    srcs = ["clientauth.go"],
    tags = ["automanaged"],
    deps = ["//pkg/client/restclient:go_default_library"],
)

go_test(
    name = "go_default_xtest",
    srcs = ["clientauth_test.go"],
    tags = ["automanaged"],
    deps = ["//pkg/client/unversioned/auth:go_default_library"],
)
