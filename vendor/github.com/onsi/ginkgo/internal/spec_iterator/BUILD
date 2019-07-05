load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "index_computer.go",
        "parallel_spec_iterator.go",
        "serial_spec_iterator.go",
        "sharded_parallel_spec_iterator.go",
        "spec_iterator.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/onsi/ginkgo/internal/spec_iterator",
    importpath = "github.com/onsi/ginkgo/internal/spec_iterator",
    visibility = ["//vendor/github.com/onsi/ginkgo:__subpackages__"],
    deps = ["//vendor/github.com/onsi/ginkgo/internal/spec:go_default_library"],
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
