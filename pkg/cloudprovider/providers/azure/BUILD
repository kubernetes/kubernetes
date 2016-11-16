package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "azure.go",
        "azure_blob.go",
        "azure_instances.go",
        "azure_loadbalancer.go",
        "azure_routes.go",
        "azure_storage.go",
        "azure_storageaccount.go",
        "azure_util.go",
        "azure_wrap.go",
        "azure_zones.go",
        "vhd.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/service:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/errors:go_default_library",
        "//vendor:github.com/Azure/azure-sdk-for-go/arm/compute",
        "//vendor:github.com/Azure/azure-sdk-for-go/arm/network",
        "//vendor:github.com/Azure/azure-sdk-for-go/arm/storage",
        "//vendor:github.com/Azure/azure-sdk-for-go/storage",
        "//vendor:github.com/Azure/go-autorest/autorest",
        "//vendor:github.com/Azure/go-autorest/autorest/azure",
        "//vendor:github.com/Azure/go-autorest/autorest/to",
        "//vendor:github.com/ghodss/yaml",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/rubiojr/go-vhd/vhd",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["azure_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/service:go_default_library",
        "//pkg/types:go_default_library",
        "//vendor:github.com/Azure/azure-sdk-for-go/arm/compute",
        "//vendor:github.com/Azure/azure-sdk-for-go/arm/network",
        "//vendor:github.com/Azure/go-autorest/autorest/to",
    ],
)
