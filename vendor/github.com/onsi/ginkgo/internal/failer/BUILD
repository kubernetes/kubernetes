load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["failer.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/onsi/ginkgo/internal/failer",
    importpath = "github.com/onsi/ginkgo/internal/failer",
    visibility = ["//vendor/github.com/onsi/ginkgo:__subpackages__"],
    deps = ["//vendor/github.com/onsi/ginkgo/types:go_default_library"],
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
