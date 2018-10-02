load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["api.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/cloudflare/cfssl/api",
    importpath = "github.com/cloudflare/cfssl/api",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/cloudflare/cfssl/errors:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/log:go_default_library",
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
        "//vendor/github.com/cloudflare/cfssl/api/bundle:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/api/certinfo:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/api/client:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/api/crl:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/api/gencrl:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/api/generator:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/api/health:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/api/info:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/api/initca:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/api/ocsp:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/api/revoke:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/api/scan:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/api/signhandler:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
