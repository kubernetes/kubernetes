package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["group_kind_visitor.go"],
    tags = ["automanaged"],
    deps = ["//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library"],
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

go_test(
    name = "go_default_xtest",
    srcs = [
        "apps_suite_test.go",
        "group_kind_visitory_test.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/kubectl/apps:go_default_library",
        "//vendor/github.com/onsi/ginkgo:go_default_library",
        "//vendor/github.com/onsi/gomega:go_default_library",
    ],
)
