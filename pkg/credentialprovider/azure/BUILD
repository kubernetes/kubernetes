package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "azure_acr_helper.go",
        "azure_credentials.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/credentialprovider/azure",
    deps = [
        "//pkg/credentialprovider:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/azure/auth:go_default_library",
        "//vendor/github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2019-05-01/containerregistry:go_default_library",
        "//vendor/github.com/Azure/go-autorest/autorest:go_default_library",
        "//vendor/github.com/Azure/go-autorest/autorest/adal:go_default_library",
        "//vendor/github.com/Azure/go-autorest/autorest/azure:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
        "//vendor/sigs.k8s.io/yaml:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["azure_credentials_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//vendor/github.com/Azure/azure-sdk-for-go/services/containerregistry/mgmt/2019-05-01/containerregistry:go_default_library",
        "//vendor/github.com/Azure/go-autorest/autorest/azure:go_default_library",
        "//vendor/github.com/Azure/go-autorest/autorest/to:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
