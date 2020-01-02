load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["kind_visitor.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kubectl/pkg/apps",
    importpath = "k8s.io/kubectl/pkg/apps",
    visibility = ["//visibility:public"],
    deps = ["//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "apps_suite_test.go",
        "kind_visitor_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//vendor/github.com/onsi/ginkgo:go_default_library",
        "//vendor/github.com/onsi/ginkgo/config:go_default_library",
        "//vendor/github.com/onsi/ginkgo/types:go_default_library",
        "//vendor/github.com/onsi/gomega:go_default_library",
    ],
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
