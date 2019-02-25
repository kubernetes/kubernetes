load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["database_accessor.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/cloudflare/cfssl/certdb/sql",
    importpath = "github.com/cloudflare/cfssl/certdb/sql",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/cloudflare/cfssl/certdb:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/errors:go_default_library",
        "//vendor/github.com/jmoiron/sqlx:go_default_library",
        "//vendor/github.com/kisielk/sqlstruct:go_default_library",
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
