load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bipartitegraph.go",
        "bipartitegraphmatching.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/onsi/gomega/matchers/support/goraph/bipartitegraph",
    importpath = "github.com/onsi/gomega/matchers/support/goraph/bipartitegraph",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/onsi/gomega/matchers/support/goraph/edge:go_default_library",
        "//vendor/github.com/onsi/gomega/matchers/support/goraph/node:go_default_library",
        "//vendor/github.com/onsi/gomega/matchers/support/goraph/util:go_default_library",
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
