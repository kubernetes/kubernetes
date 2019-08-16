load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "async.go",
        "azure.go",
        "environments.go",
        "metadata_environment.go",
        "rp.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/Azure/go-autorest/autorest/azure",
    importpath = "github.com/Azure/go-autorest/autorest/azure",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/Azure/go-autorest/autorest:go_default_library",
        "//vendor/github.com/Azure/go-autorest/tracing:go_default_library",
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
