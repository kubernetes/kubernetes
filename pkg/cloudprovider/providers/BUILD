package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "providers.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/cloudprovider/providers",
    visibility = [
        "//cmd/cloud-controller-manager:__pkg__",
        "//cmd/kube-apiserver/app/options:__pkg__",
        "//cmd/kube-controller-manager/app:__pkg__",
        "//cmd/kubelet/app:__pkg__",
    ],
    deps = [
        "//staging/src/k8s.io/legacy-cloud-providers/aws:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/azure:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/gce:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/openstack:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/vsphere:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = ["providers_test.go"],
    embed = [":go_default_library"],
    deps = ["//staging/src/k8s.io/legacy-cloud-providers/vsphere/testing:go_default_library"],
)
