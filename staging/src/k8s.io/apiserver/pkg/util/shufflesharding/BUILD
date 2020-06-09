load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["shufflesharding.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/util/shufflesharding",
    importpath = "k8s.io/apiserver/pkg/util/shufflesharding",
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["shufflesharding_test.go"],
    embed = [":go_default_library"],
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
