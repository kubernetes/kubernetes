load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["plugin.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kubectl/pkg/cmd/plugin",
    importpath = "k8s.io/kubectl/pkg/cmd/plugin",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/cli-runtime/pkg/genericclioptions:go_default_library",
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

filegroup(
    name = "testdata",
    data = glob(["testdata/**"]),
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["plugin_test.go"],
    data = glob(["testdata/**"]),
    embed = [":go_default_library"],
    deps = ["//staging/src/k8s.io/cli-runtime/pkg/genericclioptions:go_default_library"],
)
