package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "default.go",
        "interfaces.go",
    ],
    deps = ["//vendor/k8s.io/apiserver/pkg/authentication/user:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = ["default_test.go"],
    library = ":go_default_library",
    deps = ["//vendor/k8s.io/apiserver/pkg/authentication/user:go_default_library"],
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
