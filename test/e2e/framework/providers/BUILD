load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["factory.go"],
    importpath = "k8s.io/kubernetes/test/e2e/framework/providers",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
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
        "//test/e2e/framework/providers/aws:all-srcs",
        "//test/e2e/framework/providers/azure:all-srcs",
        "//test/e2e/framework/providers/gce:all-srcs",
        "//test/e2e/framework/providers/kubemark:all-srcs",
        "//test/e2e/framework/providers/openstack:all-srcs",
        "//test/e2e/framework/providers/vsphere:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
