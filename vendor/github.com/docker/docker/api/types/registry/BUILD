load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "authenticate.go",
        "registry.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/api/types/registry",
    importpath = "github.com/docker/docker/api/types/registry",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/opencontainers/image-spec/specs-go/v1:go_default_library"],
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
