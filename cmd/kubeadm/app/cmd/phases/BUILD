package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "certs.go",
        "kubeconfig.go",
        "phase.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/apis/kubeadm/v1alpha1:go_default_library",
        "//cmd/kubeadm/app/apis/kubeadm/validation:go_default_library",
        "//cmd/kubeadm/app/phases/certs:go_default_library",
        "//cmd/kubeadm/app/phases/kubeconfig:go_default_library",
        "//cmd/kubeadm/app/util:go_default_library",
        "//vendor:github.com/spf13/cobra",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apimachinery/pkg/util/validation/field",
        "//vendor:k8s.io/client-go/pkg/api",
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
