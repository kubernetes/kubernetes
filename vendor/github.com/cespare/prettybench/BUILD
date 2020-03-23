load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["prettybench.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/cespare/prettybench",
    importpath = "github.com/cespare/prettybench",
    visibility = ["//visibility:private"],
    deps = ["//vendor/golang.org/x/tools/benchmark/parse:go_default_library"],
)

go_binary(
    name = "prettybench",
    embed = [":go_default_library"],
    tags = ["manual"],
    visibility = ["//visibility:public"],
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
