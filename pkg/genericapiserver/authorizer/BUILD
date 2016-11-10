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
    srcs = [
        "builtin.go",
        "delegating.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/auth/authorizer:go_default_library",
        "//pkg/auth/authorizer/abac:go_default_library",
        "//pkg/auth/authorizer/union:go_default_library",
        "//pkg/client/clientset_generated/release_1_5/typed/authorization/v1beta1:go_default_library",
        "//pkg/controller/informers:go_default_library",
        "//plugin/pkg/auth/authorizer/rbac:go_default_library",
        "//plugin/pkg/auth/authorizer/webhook:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["authz_test.go"],
    data = [
        "//pkg/auth/authorizer/abac:example_policy",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/auth/authorizer:go_default_library",
        "//pkg/auth/user:go_default_library",
    ],
)
