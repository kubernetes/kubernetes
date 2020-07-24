load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "azure_vmsizeclient.go",
        "doc.go",
        "interface.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/legacy-cloud-providers/azure/clients/vmsizeclient",
    importpath = "k8s.io/legacy-cloud-providers/azure/clients/vmsizeclient",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/client-go/util/flowcontrol:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/azure/clients:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/azure/clients/armclient:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/azure/metrics:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/azure/retry:go_default_library",
        "//vendor/github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-12-01/compute:go_default_library",
        "//vendor/github.com/Azure/go-autorest/autorest:go_default_library",
        "//vendor/github.com/Azure/go-autorest/autorest/azure:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["azure_vmsizeclient_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/client-go/util/flowcontrol:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/azure/clients:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/azure/clients/armclient:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/azure/clients/armclient/mockarmclient:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/azure/retry:go_default_library",
        "//vendor/github.com/Azure/azure-sdk-for-go/services/compute/mgmt/2019-12-01/compute:go_default_library",
        "//vendor/github.com/Azure/go-autorest/autorest:go_default_library",
        "//vendor/github.com/Azure/go-autorest/autorest/to:go_default_library",
        "//vendor/github.com/golang/mock/gomock:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
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
        "//staging/src/k8s.io/legacy-cloud-providers/azure/clients/vmsizeclient/mockvmsizeclient:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
