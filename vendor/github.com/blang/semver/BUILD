load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "json.go",
        "range.go",
        "semver.go",
        "sort.go",
        "sql.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/blang/semver",
    importpath = "github.com/blang/semver",
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
