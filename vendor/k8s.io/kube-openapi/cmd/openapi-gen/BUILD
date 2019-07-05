load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["openapi-gen.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kube-openapi/cmd/openapi-gen",
    importpath = "k8s.io/kube-openapi/cmd/openapi-gen",
    visibility = ["//visibility:private"],
    deps = [
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
        "//vendor/k8s.io/kube-openapi/cmd/openapi-gen/args:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/generators:go_default_library",
    ],
)

go_binary(
    name = "openapi-gen",
    embed = [":go_default_library"],
    tags = ["manual"],
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
        "//vendor/k8s.io/kube-openapi/cmd/openapi-gen/args:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
