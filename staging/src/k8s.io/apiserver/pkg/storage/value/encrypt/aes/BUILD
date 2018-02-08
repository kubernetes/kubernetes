package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["aes_test.go"],
    embed = [":go_default_library"],
    importpath = "k8s.io/apiserver/pkg/storage/value/encrypt/aes",
    deps = ["//vendor/k8s.io/apiserver/pkg/storage/value:go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = ["aes.go"],
    importpath = "k8s.io/apiserver/pkg/storage/value/encrypt/aes",
    deps = ["//vendor/k8s.io/apiserver/pkg/storage/value:go_default_library"],
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
