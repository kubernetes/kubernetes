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
    srcs = ["azure_credentials.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/cloudprovider/providers/azure:go_default_library",
        "//pkg/credentialprovider:go_default_library",
        "//vendor:github.com/Azure/azure-sdk-for-go/arm/containerregistry",
        "//vendor:github.com/Azure/go-autorest/autorest/azure",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/spf13/pflag",
        "//vendor:gopkg.in/yaml.v2",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["azure_credentials_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor:github.com/Azure/azure-sdk-for-go/arm/containerregistry",
        "//vendor:github.com/Azure/go-autorest/autorest/to",
    ],
)
