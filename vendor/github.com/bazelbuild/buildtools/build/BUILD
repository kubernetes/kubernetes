load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "lex.go",
        "parse.y.go",
        "print.go",
        "quote.go",
        "rewrite.go",
        "rule.go",
        "syntax.go",
        "walk.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/bazelbuild/buildtools/build",
    importpath = "github.com/bazelbuild/buildtools/build",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/bazelbuild/buildtools/tables:go_default_library"],
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
