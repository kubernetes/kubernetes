load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["lang.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/bazelbuild/bazel-gazelle/internal/language",
    importpath = "github.com/bazelbuild/bazel-gazelle/internal/language",
    visibility = ["//vendor/github.com/bazelbuild/bazel-gazelle:__subpackages__"],
    deps = [
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/config:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/resolve:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/rule:go_default_library",
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
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/language/go:all-srcs",
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/language/proto:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
