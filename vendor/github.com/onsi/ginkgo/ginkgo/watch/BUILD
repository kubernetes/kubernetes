load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "delta.go",
        "delta_tracker.go",
        "dependencies.go",
        "package_hash.go",
        "package_hashes.go",
        "suite.go",
    ],
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/onsi/ginkgo/ginkgo/testsuite:go_default_library"],
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
