load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "callee.go",
        "imports.go",
        "map.go",
        "methodsetcache.go",
        "ui.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/tools/go/types/typeutil",
    importpath = "golang.org/x/tools/go/types/typeutil",
    visibility = ["//visibility:public"],
    deps = ["//vendor/golang.org/x/tools/go/ast/astutil:go_default_library"],
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
