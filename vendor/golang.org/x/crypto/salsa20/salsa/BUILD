load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "hsalsa20.go",
        "salsa2020_amd64.s",
        "salsa208.go",
        "salsa20_amd64.go",
        "salsa20_ref.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/golang.org/x/crypto/salsa20/salsa",
    importpath = "golang.org/x/crypto/salsa20/salsa",
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
