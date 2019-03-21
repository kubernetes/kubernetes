package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["type.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/code-generator/third_party/forked/golang/reflect",
    importpath = "k8s.io/code-generator/third_party/forked/golang/reflect",
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
