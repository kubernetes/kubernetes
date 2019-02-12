load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["kustomize.go"],
    importpath = "k8s.io/kubernetes/pkg/kubectl/cmd/kustomize",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubectl/util/i18n:go_default_library",
        "//pkg/kubectl/util/templates:go_default_library",
        "//staging/src/k8s.io/cli-runtime/pkg/genericclioptions:go_default_library",
        "//staging/src/k8s.io/cli-runtime/pkg/kustomize:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/pkg/fs:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = ["kustomize_test.go"],
    embed = [":go_default_library"],
)
