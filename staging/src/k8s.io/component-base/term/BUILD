load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["term.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/component-base/term",
    importpath = "k8s.io/component-base/term",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/moby/term:go_default_library"],
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
