package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["discovery.go"],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/discovery",
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/discovery/file:go_default_library",
        "//cmd/kubeadm/app/discovery/https:go_default_library",
        "//cmd/kubeadm/app/discovery/token:go_default_library",
        "//cmd/kubeadm/app/util/kubeconfig:go_default_library",
        "//vendor/k8s.io/client-go/tools/clientcmd/api:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["discovery_test.go"],
    embed = [":go_default_library"],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/discovery",
    deps = ["//cmd/kubeadm/app/apis/kubeadm:go_default_library"],
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
        "//cmd/kubeadm/app/discovery/file:all-srcs",
        "//cmd/kubeadm/app/discovery/https:all-srcs",
        "//cmd/kubeadm/app/discovery/token:all-srcs",
    ],
    tags = ["automanaged"],
)
