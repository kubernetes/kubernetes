load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "ip.go",
        "parse.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/utils/internal/third_party/forked/golang/net",
    importpath = "k8s.io/utils/internal/third_party/forked/golang/net",
    visibility = ["//vendor/k8s.io/utils:__subpackages__"],
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
