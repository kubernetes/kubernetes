package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "builtin.go",
        "delegating.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/auth/authorizer:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/authorization/v1beta1:go_default_library",
        "//plugin/pkg/auth/authorizer/webhook:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["authz_test.go"],
    data = [
        "//pkg/auth/authorizer/abac:example_policy",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/auth/authorizer:go_default_library",
        "//pkg/auth/user:go_default_library",
    ],
)
