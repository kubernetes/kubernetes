load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["aes.go"],
    importpath = "k8s.io/apiserver/pkg/storage/value/encrypt/aes",
    visibility = ["//visibility:public"],
    deps = ["//pkg/storage/value:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = ["aes_test.go"],
    importpath = "k8s.io/apiserver/pkg/storage/value/encrypt/aes",
    library = ":go_default_library",
    deps = ["//pkg/storage/value:go_default_library"],
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
