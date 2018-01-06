package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["token.go"],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/discovery/token",
    deps = [
        "//cmd/kubeadm/app/constants:go_default_library",
        "//cmd/kubeadm/app/util/kubeconfig:go_default_library",
        "//cmd/kubeadm/app/util/pubkeypin:go_default_library",
        "//cmd/kubeadm/app/util/token:go_default_library",
        "//pkg/bootstrap/api:go_default_library",
        "//pkg/controller/bootstrap:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/client-go/tools/clientcmd:go_default_library",
        "//vendor/k8s.io/client-go/tools/clientcmd/api:go_default_library",
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
    srcs = ["token_test.go"],
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/discovery/token",
    deps = [
        "//cmd/kubeadm/app/util/kubeconfig:go_default_library",
        "//vendor/k8s.io/client-go/tools/clientcmd/api:go_default_library",
    ],
)
