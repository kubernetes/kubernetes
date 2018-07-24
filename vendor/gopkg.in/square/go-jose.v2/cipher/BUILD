load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "cbc_hmac.go",
        "concat_kdf.go",
        "ecdh_es.go",
        "key_wrap.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/gopkg.in/square/go-jose.v2/cipher",
    importpath = "gopkg.in/square/go-jose.v2/cipher",
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
