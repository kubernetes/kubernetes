load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["file.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/spf13/afero/sftp",
    importpath = "github.com/spf13/afero/sftp",
    visibility = ["//visibility:public"],
    deps = ["//vendor/github.com/pkg/sftp:go_default_library"],
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
