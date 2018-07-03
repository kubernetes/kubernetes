load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "construct.go",
        "doc.go",
        "generator.go",
        "sort_labels.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/bazelbuild/bazel-gazelle/internal/generator",
    importpath = "github.com/bazelbuild/bazel-gazelle/internal/generator",
    visibility = ["//vendor/github.com/bazelbuild/bazel-gazelle:__subpackages__"],
    deps = [
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/config:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/label:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/packages:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/pathtools:go_default_library",
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
