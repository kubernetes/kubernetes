package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["azure_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//vendor/github.com/Azure/go-autorest/autorest/azure:go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = ["azure.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/Azure/go-autorest/autorest:go_default_library",
        "//vendor/github.com/Azure/go-autorest/autorest/azure:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
    ],
)
