load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "directives.go",
        "expr.go",
        "merge.go",
        "platform.go",
        "platform_strings.go",
        "rule.go",
        "sort_labels.go",
        "types.go",
        "value.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/bazelbuild/bazel-gazelle/rule",
    importpath = "github.com/bazelbuild/bazel-gazelle/rule",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/bazelbuild/bazel-gazelle/label:go_default_library",
        "//vendor/github.com/bazelbuild/buildtools/build:go_default_library",
        "//vendor/github.com/bazelbuild/buildtools/tables:go_default_library",
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
