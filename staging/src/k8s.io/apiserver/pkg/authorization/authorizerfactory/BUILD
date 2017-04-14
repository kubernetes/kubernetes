package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["authz_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "builtin.go",
        "delegating.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//vendor/k8s.io/apiserver/plugin/pkg/authorizer/webhook:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/authorization/v1beta1:go_default_library",
    ],
)
