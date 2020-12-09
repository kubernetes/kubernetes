load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["objectid.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kube-openapi/pkg/validation/strfmt/bson",
    importpath = "k8s.io/kube-openapi/pkg/validation/strfmt/bson",
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
