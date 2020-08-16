load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "curve25519.go",
        "curve25519_amd64.go",
        "curve25519_amd64.s",
        "curve25519_generic.go",
        "curve25519_noasm.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/crypto/curve25519",
    importpath = "golang.org/x/crypto/curve25519",
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
