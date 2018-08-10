load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "bind.go",
        "doc.go",
        "named.go",
        "named_context.go",
        "sqlx.go",
        "sqlx_context.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/jmoiron/sqlx",
    importpath = "github.com/jmoiron/sqlx",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/jmoiron/sqlx/reflectx:go_default_library"],
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
        "//vendor/github.com/jmoiron/sqlx/reflectx:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
