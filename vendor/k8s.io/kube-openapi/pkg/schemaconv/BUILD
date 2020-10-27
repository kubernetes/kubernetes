load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["smd.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kube-openapi/pkg/schemaconv",
    importpath = "k8s.io/kube-openapi/pkg/schemaconv",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/k8s.io/kube-openapi/pkg/util/proto:go_default_library",
        "//vendor/sigs.k8s.io/structured-merge-diff/v4/schema:go_default_library",
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
