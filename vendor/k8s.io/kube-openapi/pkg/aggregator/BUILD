load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["aggregator.go"],
    importpath = "k8s.io/kube-openapi/pkg/aggregator",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/go-openapi/spec:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/util:go_default_library",
    ],
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
