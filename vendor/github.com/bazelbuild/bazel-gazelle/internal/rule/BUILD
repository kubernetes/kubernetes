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
    importmap = "k8s.io/kubernetes/vendor/github.com/bazelbuild/bazel-gazelle/internal/rule",
    importpath = "github.com/bazelbuild/bazel-gazelle/internal/rule",
    visibility = ["//vendor/github.com/bazelbuild/bazel-gazelle:__subpackages__"],
    deps = [
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/label:go_default_library",
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
