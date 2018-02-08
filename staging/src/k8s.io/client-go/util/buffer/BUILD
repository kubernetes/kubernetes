load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["ring_growing.go"],
    importpath = "k8s.io/client-go/util/buffer",
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["ring_growing_test.go"],
    embed = [":go_default_library"],
    importpath = "k8s.io/client-go/util/buffer",
    deps = ["//vendor/github.com/stretchr/testify/assert:go_default_library"],
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
    visibility = ["//visibility:public"],
)
