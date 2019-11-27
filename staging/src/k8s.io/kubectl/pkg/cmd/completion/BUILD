load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["completion.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kubectl/pkg/cmd/completion",
    importpath = "k8s.io/kubectl/pkg/cmd/completion",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/kubectl/pkg/cmd/util:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/util/i18n:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/util/templates:go_default_library",
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
