load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "algorithm.go",
        "digest.go",
        "digester.go",
        "doc.go",
        "verifiers.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/opencontainers/go-digest",
    importpath = "github.com/opencontainers/go-digest",
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
