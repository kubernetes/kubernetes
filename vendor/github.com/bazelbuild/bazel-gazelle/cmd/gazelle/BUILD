load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "diff.go",
        "fix.go",
        "fix-update.go",
        "gazelle.go",
        "langs.go",
        "print.go",
        "update-repos.go",
        "version.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/bazelbuild/bazel-gazelle/cmd/gazelle",
    importpath = "github.com/bazelbuild/bazel-gazelle/cmd/gazelle",
    visibility = ["//visibility:private"],
    deps = [
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/config:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/flag:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/label:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/language:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/language/go:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/language/proto:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/merger:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/repos:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/resolve:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/rule:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/version:go_default_library",
        "//vendor/github.com/bazelbuild/bazel-gazelle/internal/walk:go_default_library",
        "//vendor/github.com/pmezard/go-difflib/difflib:go_default_library",
    ],
)

go_binary(
    name = "gazelle",
    embed = [":go_default_library"],
    tags = ["manual"],
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
