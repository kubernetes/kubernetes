load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "volume_create.go",
        "volume_list.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/docker/docker/api/types/volume",
    importpath = "github.com/docker/docker/api/types/volume",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/docker/docker/api/types:go_default_library"],
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
