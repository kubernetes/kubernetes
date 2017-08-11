package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["clientauth.go"],
    deps = ["//vendor/k8s.io/client-go/rest:go_default_library"],
)

go_test(
    name = "go_default_xtest",
    srcs = ["clientauth_test.go"],
    deps = ["//vendor/k8s.io/client-go/tools/auth:go_default_library"],
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
