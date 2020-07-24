load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "azure_fileclient.go",
        "doc.go",
        "interface.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/legacy-cloud-providers/azure/clients/fileclient",
    importpath = "k8s.io/legacy-cloud-providers/azure/clients/fileclient",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/legacy-cloud-providers/azure/clients:go_default_library",
        "//vendor/github.com/Azure/azure-sdk-for-go/services/storage/mgmt/2019-06-01/storage:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
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
        "//staging/src/k8s.io/legacy-cloud-providers/azure/clients/fileclient/mockfileclient:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
