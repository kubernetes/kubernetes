load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "collection.go",
        "constraints.go",
        "doc.go",
        "error.go",
        "magic.go",
        "parse.go",
        "range.go",
        "union.go",
        "version.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/Masterminds/semver",
    importpath = "github.com/Masterminds/semver",
    visibility = ["//visibility:public"],
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
