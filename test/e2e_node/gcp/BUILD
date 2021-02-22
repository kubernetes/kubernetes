load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["framework.go"],
    importpath = "k8s.io/kubernetes/test/e2e_node/gcp",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/onsi/ginkgo:go_default_library"],
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

go_test(
    name = "go_default_test",
    srcs = ["gke_environment_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//test/e2e/framework:go_default_library",
        "//test/e2e/framework/skipper:go_default_library",
        "//test/utils/image:go_default_library",
        "//vendor/github.com/blang/semver:go_default_library",
        "//vendor/github.com/onsi/ginkgo:go_default_library",
    ],
)
