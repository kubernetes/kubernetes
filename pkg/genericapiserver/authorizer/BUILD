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
        "//pkg/client/clientset_generated/clientset/typed/authorization/v1beta1:go_default_library",
        "//plugin/pkg/auth/authorizer/webhook:go_default_library",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
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
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
    ],
)
