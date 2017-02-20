package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["controllermanager.go"],
    tags = ["automanaged"],
    deps = [
        "//cmd/cloud-controller-manager/app/options:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions:go_default_library",
        "//pkg/client/leaderelection:go_default_library",
        "//pkg/client/leaderelection/resourcelock:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/controller:go_default_library",
        "//pkg/controller/cloud:go_default_library",
        "//pkg/controller/informers:go_default_library",
        "//pkg/controller/route:go_default_library",
        "//pkg/controller/service:go_default_library",
        "//pkg/util/configz:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
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
        "//cmd/cloud-controller-manager/app/options:all-srcs",
    ],
    tags = ["automanaged"],
)
