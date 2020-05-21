load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "discovery.go",
        "urls.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/library-go/pkg/oauth/oauthdiscovery",
    importpath = "github.com/openshift/library-go/pkg/oauth/oauthdiscovery",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
