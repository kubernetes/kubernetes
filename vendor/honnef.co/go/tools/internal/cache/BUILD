load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "cache.go",
        "default.go",
        "hash.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/honnef.co/go/tools/internal/cache",
    importpath = "honnef.co/go/tools/internal/cache",
    visibility = ["//vendor/honnef.co/go/tools:__subpackages__"],
    deps = ["//vendor/honnef.co/go/tools/internal/renameio:go_default_library"],
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
