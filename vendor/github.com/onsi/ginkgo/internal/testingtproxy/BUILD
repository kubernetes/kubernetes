load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["testing_t_proxy.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/onsi/ginkgo/internal/testingtproxy",
    importpath = "github.com/onsi/ginkgo/internal/testingtproxy",
    visibility = ["//vendor/github.com/onsi/ginkgo:__subpackages__"],
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
