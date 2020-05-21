load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["interface.go"],
    importmap = "k8s.io/kubernetes/vendor/github.com/openshift/client-go/user/informers/externalversions/user",
    importpath = "github.com/openshift/client-go/user/informers/externalversions/user",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/openshift/client-go/user/informers/externalversions/internalinterfaces:go_default_library",
        "//vendor/github.com/openshift/client-go/user/informers/externalversions/user/v1:go_default_library",
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
        "//vendor/github.com/openshift/client-go/user/informers/externalversions/user/v1:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
