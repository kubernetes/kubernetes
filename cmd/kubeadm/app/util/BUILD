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
        "error.go",
        "kubeconfig.go",
        "tokens.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/preflight:go_default_library",
        "//pkg/client/unversioned/clientcmd:go_default_library",
        "//pkg/client/unversioned/clientcmd/api:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/renstrom/dedent",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "error_test.go",
        "kubeconfig_test.go",
        "tokens_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/preflight:go_default_library",
        "//pkg/client/unversioned/clientcmd/api:go_default_library",
    ],
)
