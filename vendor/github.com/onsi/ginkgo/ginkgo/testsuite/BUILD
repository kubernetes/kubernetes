load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "test_suite.go",
        "vendor_check_go15.go",
        "vendor_check_go16.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/onsi/ginkgo/ginkgo/testsuite",
    importpath = "github.com/onsi/ginkgo/ginkgo/testsuite",
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
