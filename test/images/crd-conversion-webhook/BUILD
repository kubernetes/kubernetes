load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "config.go",
        "main.go",
    ],
    importpath = "k8s.io/kubernetes/test/images/crd-conversion-webhook",
    visibility = ["//visibility:private"],
    deps = [
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//test/images/crd-conversion-webhook/converter:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ],
)

go_binary(
    name = "crd-conversion-webhook",
    embed = [":go_default_library"],
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
    srcs = [
        ":package-srcs",
        "//test/images/crd-conversion-webhook/converter:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
