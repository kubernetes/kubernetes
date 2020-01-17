load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["renameio.go"],
    importmap = "k8s.io/kubernetes/vendor/honnef.co/go/tools/internal/renameio",
    importpath = "honnef.co/go/tools/internal/renameio",
    visibility = ["//vendor/honnef.co/go/tools:__subpackages__"],
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
