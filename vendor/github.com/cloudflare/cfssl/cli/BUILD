load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "cli.go",
        "config.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/cloudflare/cfssl/cli",
    importpath = "github.com/cloudflare/cfssl/cli",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/cloudflare/cfssl/config:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/helpers:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/log:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/signer/universal:go_default_library",
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
        "//vendor/github.com/cloudflare/cfssl/cli/bundle:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/cli/certinfo:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/cli/crl:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/cli/gencert:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/cli/gencrl:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/cli/gencsr:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/cli/genkey:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/cli/info:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/cli/ocspdump:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/cli/ocsprefresh:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/cli/ocspserve:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/cli/ocspsign:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/cli/printdefault:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/cli/revoke:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/cli/scan:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/cli/selfsign:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/cli/serve:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/cli/sign:all-srcs",
        "//vendor/github.com/cloudflare/cfssl/cli/version:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
