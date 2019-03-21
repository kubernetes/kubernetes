load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["main.go"],
    importpath = "k8s.io/kubernetes/test/images/audit-proxy",
    visibility = ["//visibility:private"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer/json:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/apis/audit/install:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/apis/audit/v1:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/audit:go_default_library",
    ],
)

go_binary(
    name = "audit-proxy",
    embed = [":go_default_library"],
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
