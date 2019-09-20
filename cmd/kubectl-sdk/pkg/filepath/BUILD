load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["filepath.go"],
    importpath = "k8s.io/kubernetes/cmd/kubectl-sdk/pkg/filepath",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/kubectl-sdk/pkg/util:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/version:go_default_library",
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
