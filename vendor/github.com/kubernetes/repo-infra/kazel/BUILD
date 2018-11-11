load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "config.go",
        "diff.go",
        "generator.go",
        "kazel.go",
        "sourcerer.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/kubernetes/repo-infra/kazel",
    importpath = "github.com/kubernetes/repo-infra/kazel",
    visibility = ["//visibility:private"],
    deps = [
        "//vendor/github.com/bazelbuild/buildtools/build:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ],
)

go_binary(
    name = "kazel",
    embed = [":go_default_library"],
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
