package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["clientauth.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/client-go/tools/auth",
    importpath = "k8s.io/client-go/tools/auth",
    deps = ["//staging/src/k8s.io/client-go/rest:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = ["clientauth_test.go"],
    embed = [":go_default_library"],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//staging/src/k8s.io/client-go/tools/auth/exec:all-srcs",
    ],
    tags = ["automanaged"],
)
