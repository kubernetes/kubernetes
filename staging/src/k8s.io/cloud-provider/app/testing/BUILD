load("@io_bazel_rules_go//go:def.bzl", "go_library")

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

go_library(
    name = "go_default_library",
    srcs = ["testserver.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/cloud-provider/app/testing",
    importpath = "k8s.io/cloud-provider/app/testing",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/cloud-provider:go_default_library",
        "//staging/src/k8s.io/cloud-provider/app:go_default_library",
        "//staging/src/k8s.io/cloud-provider/app/config:go_default_library",
        "//staging/src/k8s.io/cloud-provider/options:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
    ],
)
