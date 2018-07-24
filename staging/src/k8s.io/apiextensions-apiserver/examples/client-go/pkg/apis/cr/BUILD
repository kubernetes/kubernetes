load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["register.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiextensions-apiserver/examples/client-go/pkg/apis/cr",
    importpath = "k8s.io/apiextensions-apiserver/examples/client-go/pkg/apis/cr",
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
        "//staging/src/k8s.io/apiextensions-apiserver/examples/client-go/pkg/apis/cr/v1:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
