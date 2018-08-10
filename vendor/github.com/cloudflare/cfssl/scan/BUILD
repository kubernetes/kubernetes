load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "broad.go",
        "connectivity.go",
        "pki.go",
        "scan_common.go",
        "tls_handshake.go",
        "tls_session.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/cloudflare/cfssl/scan",
    importpath = "github.com/cloudflare/cfssl/scan",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/cloudflare/cfssl/bundler:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/helpers:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/log:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/revoke:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/scan/crypto/tls:go_default_library",
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
        "//vendor/github.com/cloudflare/cfssl/scan/crypto/tls:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
