load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "config.go",
        "constants.go",
        "fileinfo.go",
        "fix.go",
        "generate.go",
        "kinds.go",
        "known_imports.go",
        "lang.go",
        "package.go",
        "resolve.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/bazelbuild/bazel-gazelle/language/proto",
    importpath = "github.com/bazelbuild/bazel-gazelle/language/proto",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/bazelbuild/bazel-gazelle/config:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/label:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/language:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/repo:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/resolve:go_default_library",
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
