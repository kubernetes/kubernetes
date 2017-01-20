package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["config_test.go"],
    data = [
        "//pkg/auth/authorizer/abac:example_policy",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
)

go_library(
    name = "go_default_library",
    srcs = ["config.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/auth/authorizer/abac:go_default_library",
        "//pkg/controller/informers:go_default_library",
        "//pkg/genericapiserver/authorizer:go_default_library",
        "//plugin/pkg/auth/authorizer/rbac:go_default_library",
        "//plugin/pkg/auth/authorizer/webhook:go_default_library",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
        "//vendor:k8s.io/apiserver/pkg/authorization/union",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
