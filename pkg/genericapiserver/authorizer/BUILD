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
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
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
        "//plugin/pkg/auth/authorizer/webhook:go_default_library",
        "//vendor:k8s.io/apiserver/pkg/authorization/authorizer",
        "//vendor:k8s.io/client-go/kubernetes/typed/authorization/v1beta1",
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
