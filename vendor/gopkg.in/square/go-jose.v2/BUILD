load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "asymmetric.go",
        "crypter.go",
        "doc.go",
        "encoding.go",
        "jwe.go",
        "jwk.go",
        "jws.go",
        "shared.go",
        "signing.go",
        "symmetric.go",
    ],
    importpath = "gopkg.in/square/go-jose.v2",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/golang.org/x/crypto/ed25519:go_default_library",
        "//vendor/gopkg.in/square/go-jose.v2/cipher:go_default_library",
        "//vendor/gopkg.in/square/go-jose.v2/json:go_default_library",
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
        "//vendor/gopkg.in/square/go-jose.v2/cipher:all-srcs",
        "//vendor/gopkg.in/square/go-jose.v2/json:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
