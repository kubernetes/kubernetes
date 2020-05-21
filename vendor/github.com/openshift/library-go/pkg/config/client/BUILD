load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["client_config.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/library-go/pkg/config/client",
    importpath = "github.com/openshift/library-go/pkg/config/client",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/tools/clientcmd:go_default_library",
        "//vendor/github.com/openshift/api/config/v1:go_default_library",
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
