load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["signer.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/cloudflare/cfssl/signer",
    importpath = "github.com/cloudflare/cfssl/signer",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/cloudflare/cfssl/certdb:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/config:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/csr:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/errors:go_default_library",
        "//vendor/github.com/cloudflare/cfssl/info:go_default_library",
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
        "//vendor/github.com/cloudflare/cfssl/signer/local:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
