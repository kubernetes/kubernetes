package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["helpers.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/authorization:go_default_library",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
    ],
)
