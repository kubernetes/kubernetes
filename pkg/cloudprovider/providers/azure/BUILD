package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
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
        "//pkg/api/v1/service:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/volume:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["azure_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//pkg/api/v1/service:go_default_library"],
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
