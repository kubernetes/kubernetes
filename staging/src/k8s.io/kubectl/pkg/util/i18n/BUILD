load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["i18n.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kubectl/pkg/util/i18n",
    importpath = "k8s.io/kubectl/pkg/util/i18n",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/kubectl/pkg/generated:go_default_library",
        "//vendor/github.com/chai2010/gettext-go/gettext:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["i18n_test.go"],
    embed = [":go_default_library"],
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
