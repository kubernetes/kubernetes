load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["help.go"],
    importpath = "k8s.io/kubernetes/pkg/kubectl/cmd/help",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubectl/util/i18n:go_default_library",
        "//pkg/kubectl/util/templates:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
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
