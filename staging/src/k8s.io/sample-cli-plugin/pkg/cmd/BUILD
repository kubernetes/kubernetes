load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["ns.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/sample-cli-plugin/pkg/cmd",
    importpath = "k8s.io/sample-cli-plugin/pkg/cmd",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/cli-runtime/pkg/genericclioptions:go_default_library",
        "//staging/src/k8s.io/client-go/tools/clientcmd:go_default_library",
        "//staging/src/k8s.io/client-go/tools/clientcmd/api:go_default_library",
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
