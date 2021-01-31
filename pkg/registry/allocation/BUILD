load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["ipallocator.go"],
    importpath = "k8s.io/kubernetes/pkg/registry/allocation",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/registry/core/service/ipallocator:go_default_library",
        "//staging/src/k8s.io/api/allocation/v1alpha1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/typed/allocation/v1alpha1:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
        "//vendor/k8s.io/utils/net:go_default_library",
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
        "//pkg/registry/allocation/ipaddress:all-srcs",
        "//pkg/registry/allocation/iprange:all-srcs",
        "//pkg/registry/allocation/rest:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
