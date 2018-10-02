load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["serve.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/cloudflare/cfssl/cli/serve",
    importpath = "github.com/cloudflare/cfssl/cli/serve",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/GeertJohan/go.rice:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/api:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/api/bundle:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/api/certinfo:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/api/crl:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/api/gencrl:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/api/generator:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/api/health:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/api/info:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/api/initca:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/api/ocsp:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/api/revoke:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/api/scan:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/api/signhandler:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/bundler:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/certdb/dbconf:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/certdb/sql:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/cli:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/cli/ocspsign:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/cli/sign:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/helpers:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/log:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/ocsp:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/signer:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/ubiquity:go_default_library",
        "//vendor/github.com/jmoiron/sqlx:go_default_library",
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
