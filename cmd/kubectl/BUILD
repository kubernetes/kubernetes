load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)
load("//staging/src/k8s.io/component-base/version:def.bzl", "version_x_defs")

go_binary(
    name = "kubectl",
    embed = [":go_default_library"],
    pure = "on",
    visibility = ["//visibility:public"],
    x_defs = version_x_defs(),
)

go_library(
    name = "go_default_library",
    srcs = ["kubectl.go"],
    importpath = "k8s.io/kubernetes/cmd/kubectl",
    visibility = ["//visibility:private"],
    deps = [
        "//pkg/kubectl/cmd:go_default_library",
        "//staging/src/k8s.io/client-go/plugin/pkg/client/auth:go_default_library",
        "//staging/src/k8s.io/component-base/cli/flag:go_default_library",
        "//staging/src/k8s.io/kubectl/pkg/util/logs:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = [
        "//build/visible_to:cmd_kubectl_CONSUMERS",
    ],
)
