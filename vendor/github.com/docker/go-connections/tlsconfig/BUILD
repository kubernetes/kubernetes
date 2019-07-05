load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "certpool_go17.go",
        "certpool_other.go",
        "config.go",
        "config_client_ciphers.go",
        "config_legacy_client_ciphers.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/go-connections/tlsconfig",
    importpath = "github.com/docker/go-connections/tlsconfig",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/pkg/errors:go_default_library"],
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
