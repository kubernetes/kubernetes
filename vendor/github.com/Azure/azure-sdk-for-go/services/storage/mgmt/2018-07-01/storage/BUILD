load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "accounts.go",
        "blobcontainers.go",
        "blobservices.go",
        "client.go",
        "managementpolicies.go",
        "models.go",
        "operations.go",
        "skus.go",
        "usages.go",
        "version.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2018-07-01/storage",
    importpath = "github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2018-07-01/storage",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/Azure/azure-sdk-for-go/version:go_default_library",
        "//vendor/github.com/Azure/go-autorest/autorest:go_default_library",
        "//vendor/github.com/Azure/go-autorest/autorest/azure:go_default_library",
        "//vendor/github.com/Azure/go-autorest/autorest/date:go_default_library",
        "//vendor/github.com/Azure/go-autorest/autorest/validation:go_default_library",
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
