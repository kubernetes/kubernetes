load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "default_reporter.go",
        "fake_reporter.go",
        "junit_reporter.go",
        "reporter.go",
        "teamcity_reporter.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/onsi/ginkgo/config:go_default_library",
        "//vendor/github.com/onsi/ginkgo/reporters/stenographer:go_default_library",
        "//vendor/github.com/onsi/ginkgo/types:go_default_library",
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
        "//vendor/github.com/onsi/ginkgo/reporters/stenographer:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
