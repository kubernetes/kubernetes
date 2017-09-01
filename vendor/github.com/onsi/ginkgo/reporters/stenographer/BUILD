load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "console_logging.go",
        "fake_stenographer.go",
        "stenographer.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/onsi/ginkgo/reporters/stenographer/support/go-colorable:go_default_library",
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
        "//vendor/github.com/onsi/ginkgo/reporters/stenographer/support/go-colorable:all-srcs",
        "//vendor/github.com/onsi/ginkgo/reporters/stenographer/support/go-isatty:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
