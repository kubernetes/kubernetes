package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "cinder_provider.go",
        "doc.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/volume/cinder",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/cloud-provider:go_default_library",
        "//vendor/k8s.io/cloud-provider-openstack/pkg/cloudprovider/providers/openstack:go_default_library",
        "//vendor/k8s.io/cloud-provider-openstack/pkg/volume/cinder:go_default_library",
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
