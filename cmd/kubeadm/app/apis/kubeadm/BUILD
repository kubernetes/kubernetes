package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "env.go",
        "register.go",
        "types.go",
        "well_known_labels.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
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
        "//cmd/kubeadm/app/apis/kubeadm/fuzzer:all-srcs",
        "//cmd/kubeadm/app/apis/kubeadm/install:all-srcs",
        "//cmd/kubeadm/app/apis/kubeadm/v1alpha1:all-srcs",
        "//cmd/kubeadm/app/apis/kubeadm/validation:all-srcs",
    ],
    tags = ["automanaged"],
)
