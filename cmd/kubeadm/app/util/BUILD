package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "error.go",
        "kubeconfig.go",
        "tokens.go",
        "version.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/preflight:go_default_library",
        "//pkg/client/unversioned/clientcmd:go_default_library",
        "//pkg/client/unversioned/clientcmd/api:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "error_test.go",
        "kubeconfig_test.go",
        "tokens_test.go",
        "version_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/preflight:go_default_library",
        "//pkg/client/unversioned/clientcmd/api:go_default_library",
    ],
)
