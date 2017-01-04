package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["escalation_check.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/genericapiserver/api/request:go_default_library",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
    ],
)
