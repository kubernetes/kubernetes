load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "gomega_dsl.go",
        "matchers.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/onsi/gomega",
    importpath = "github.com/onsi/gomega",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/onsi/gomega/internal/assertion:go_default_library",
        "//vendor/github.com/onsi/gomega/internal/asyncassertion:go_default_library",
        "//vendor/github.com/onsi/gomega/internal/testingtsupport:go_default_library",
        "//vendor/github.com/onsi/gomega/matchers:go_default_library",
        "//vendor/github.com/onsi/gomega/types:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/github.com/onsi/gomega/format:all-srcs",
        "//vendor/github.com/onsi/gomega/gstruct:all-srcs",
        "//vendor/github.com/onsi/gomega/internal/assertion:all-srcs",
        "//vendor/github.com/onsi/gomega/internal/asyncassertion:all-srcs",
        "//vendor/github.com/onsi/gomega/internal/oraclematcher:all-srcs",
        "//vendor/github.com/onsi/gomega/internal/testingtsupport:all-srcs",
        "//vendor/github.com/onsi/gomega/matchers:all-srcs",
        "//vendor/github.com/onsi/gomega/types:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
