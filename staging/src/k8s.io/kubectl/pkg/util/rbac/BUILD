load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["rbac.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kubectl/pkg/util/rbac",
    importpath = "k8s.io/kubectl/pkg/util/rbac",
    visibility = ["//visibility:public"],
    deps = ["//staging/src/k8s.io/api/rbac/v1:go_default_library"],
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
