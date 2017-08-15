package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "manifests_test.go",
        "volumes_test.go",
    ],
    library = ":go_default_library",
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/constants:go_default_library",
        "//cmd/kubeadm/test:go_default_library",
        "//pkg/util/version:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "manifests.go",
        "volumes.go",
    ],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/apis/kubeadm/v1alpha1:go_default_library",
        "//cmd/kubeadm/app/constants:go_default_library",
        "//cmd/kubeadm/app/images:go_default_library",
        "//cmd/kubeadm/app/util/staticpod:go_default_library",
        "//pkg/kubeapiserver/authorizer/modes:go_default_library",
        "//pkg/util/version:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
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
