load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "enclosing.go",
        "imports.go",
        "rewrite.go",
        "util.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/tools/go/ast/astutil",
    importpath = "golang.org/x/tools/go/ast/astutil",
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
