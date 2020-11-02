load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "helpers.go",
        "normalize.go",
        "reference.go",
        "regexp.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/distribution/reference",
    importpath = "github.com/docker/distribution/reference",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/docker/distribution/digestset:go_default_library",
        "//vendor/github.com/opencontainers/go-digest:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
