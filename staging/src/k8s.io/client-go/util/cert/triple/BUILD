package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["triple.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/client-go/util/cert/triple",
    importpath = "k8s.io/client-go/util/cert/triple",
    deps = ["//staging/src/k8s.io/client-go/util/cert:go_default_library"],
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
