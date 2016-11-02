package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "bootstrap.go",
        "csr.go",
        "discovery.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/util:go_default_library",
        "//pkg/apis/certificates:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/certificates/internalversion:go_default_library",
        "//pkg/client/unversioned/clientcmd:go_default_library",
        "//pkg/client/unversioned/clientcmd/api:go_default_library",
        "//pkg/kubelet/util/csr:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/cert:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//vendor:github.com/square/go-jose",
    ],
)
