load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "ansi_others.go",
        "ansi_windows.go",
        "colors.go",
        "no_colors.go",
        "writer.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/DATA-DOG/godog/colors",
    importpath = "github.com/DATA-DOG/godog/colors",
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
