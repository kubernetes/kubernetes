load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["apply.go"],
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubectl:go_default_library",
        "//pkg/kubectl/apply/merge:go_default_library",
        "//pkg/kubectl/apply/parse:go_default_library",
        "//pkg/kubectl/cmd/templates:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/kubectl/resource:go_default_library",
        "//pkg/kubectl/util/i18n:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/json:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/mergepatch:go_default_library",
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
