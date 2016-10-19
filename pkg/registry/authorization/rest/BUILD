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
    srcs = ["storage_authorization.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/rest:go_default_library",
        "//pkg/apis/authorization:go_default_library",
        "//pkg/apis/authorization/v1beta1:go_default_library",
        "//pkg/auth/authorizer:go_default_library",
        "//pkg/genericapiserver:go_default_library",
        "//pkg/registry/authorization/localsubjectaccessreview:go_default_library",
        "//pkg/registry/authorization/selfsubjectaccessreview:go_default_library",
        "//pkg/registry/authorization/subjectaccessreview:go_default_library",
    ],
)
