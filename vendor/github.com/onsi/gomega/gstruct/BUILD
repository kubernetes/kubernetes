load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "elements.go",
        "fields.go",
        "ignore.go",
        "pointer.go",
        "types.go",
    ],
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/onsi/gomega/format:go_default_library",
        "//vendor/github.com/onsi/gomega/gstruct/errors:go_default_library",
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
        "//vendor/github.com/onsi/gomega/gstruct/errors:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
