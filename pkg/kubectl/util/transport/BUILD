load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["round_tripper.go"],
    importpath = "k8s.io/kubernetes/pkg/kubectl/util/transport",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/gregjones/httpcache:go_default_library",
        "//vendor/github.com/gregjones/httpcache/diskcache:go_default_library",
        "//vendor/github.com/peterbourgon/diskv:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["round_tripper_test.go"],
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/pkg/kubectl/util/transport",
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
