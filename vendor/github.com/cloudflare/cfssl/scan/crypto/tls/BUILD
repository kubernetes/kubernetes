load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "alert.go",
        "cfsslscan_common.go",
        "cfsslscan_handshake.go",
        "cipher_suites.go",
        "common.go",
        "conn.go",
        "handshake_client.go",
        "handshake_messages.go",
        "handshake_server.go",
        "key_agreement.go",
        "prf.go",
        "ticket.go",
        "tls.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/cloudflare/cfssl/scan/crypto/tls",
    importpath = "github.com/cloudflare/cfssl/scan/crypto/tls",
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
