package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["union_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//pkg/auth/authorizer:go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = ["union.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/auth/authorizer:go_default_library",
        "//vendor:k8s.io/client-go/pkg/util/errors",
    ],
)
