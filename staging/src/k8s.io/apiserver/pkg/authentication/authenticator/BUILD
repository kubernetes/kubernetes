package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["interfaces.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/authentication/authenticator",
    importpath = "k8s.io/apiserver/pkg/authentication/authenticator",
    deps = ["//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library"],
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
