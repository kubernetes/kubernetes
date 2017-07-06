licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "umask.go",
        "util.go",
    ],
    tags = ["automanaged"],
    visibility = ["//build/visible_to:pkg_kubectl_util_CONSUMERS"],
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//pkg/kubectl/util/crlf:all-srcs",
        "//pkg/kubectl/util/slice:all-srcs",
        "//pkg/kubectl/util/term:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//build/visible_to:pkg_kubectl_util_CONSUMERS"],
)
