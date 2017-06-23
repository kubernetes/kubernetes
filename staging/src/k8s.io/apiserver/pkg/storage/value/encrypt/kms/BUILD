package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["gkms.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/cloudprovider:go_default_library",
        "//pkg/cloudprovider/providers/gce:go_default_library",
        "//pkg/kubeapiserver/options:go_default_library",
        "//vendor/google.golang.org/api/cloudkms/v1:go_default_library",
        "//vendor/google.golang.org/api/googleapi:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/storage/value:go_default_library",
    ],
)
