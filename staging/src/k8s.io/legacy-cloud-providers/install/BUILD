package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "providers.go",
        "test_package.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/legacy-cloud-providers/install",
    importpath = "k8s.io/legacy-cloud-providers/install",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/legacy-cloud-providers/aws:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/azure:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/gce:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/openstack:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/vsphere:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/vsphere/testing:go_default_library",
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
    visibility = ["//visibility:public"],
)
