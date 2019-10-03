package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["latest.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/internal-api/apis/abac/latest",
    importpath = "k8s.io/internal-api/apis/abac/latest",
    deps = [
        "//staging/src/k8s.io/internal-api/apis/abac:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/abac/v0:go_default_library",
        "//staging/src/k8s.io/internal-api/apis/abac/v1beta1:go_default_library",
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
)
