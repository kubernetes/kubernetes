load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "ast.go",
        "astbuilder.go",
        "dialect.go",
        "dialects_builtin.go",
        "gherkin.go",
        "matcher.go",
        "parser.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/DATA-DOG/godog/gherkin",
    importpath = "github.com/DATA-DOG/godog/gherkin",
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
