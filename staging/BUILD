package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//staging/src/k8s.io/api:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver:all-srcs",
        "//staging/src/k8s.io/apimachinery:all-srcs",
        "//staging/src/k8s.io/apiserver:all-srcs",
        "//staging/src/k8s.io/client-go:all-srcs",
        "//staging/src/k8s.io/code-generator:all-srcs",
        "//staging/src/k8s.io/kube-aggregator:all-srcs",
        "//staging/src/k8s.io/metrics:all-srcs",
        "//staging/src/k8s.io/sample-apiserver:all-srcs",
        "//staging/src/k8s.io/sample-controller:all-srcs",
    ],
    tags = ["automanaged"],
)
