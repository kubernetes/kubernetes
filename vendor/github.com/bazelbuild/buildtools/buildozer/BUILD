load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["main.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/bazelbuild/buildtools/buildozer",
    importpath = "github.com/bazelbuild/buildtools/buildozer",
    visibility = ["//visibility:private"],
    deps = [
        "//vendor/github.com/bazelbuild/buildtools/edit:go_default_library",
        "//vendor/github.com/bazelbuild/buildtools/tables:go_default_library",
    ],
)

go_binary(
    name = "buildozer",
    embed = [":go_default_library"],
    tags = ["manual"],
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
