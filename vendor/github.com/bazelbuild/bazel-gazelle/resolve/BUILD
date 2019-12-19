load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "config.go",
        "index.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/bazelbuild/bazel-gazelle/resolve",
    importpath = "github.com/bazelbuild/bazel-gazelle/resolve",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/bazelbuild/bazel-gazelle/config:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/label:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/repo:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/rule:go_default_library",
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
