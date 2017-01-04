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
        "discovery.go",
        "flags.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/discovery/file:go_default_library",
        "//cmd/kubeadm/app/discovery/https:go_default_library",
        "//cmd/kubeadm/app/discovery/token:go_default_library",
        "//pkg/client/unversioned/clientcmd/api:go_default_library",
        "//vendor:github.com/spf13/pflag",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["flags_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//vendor:github.com/davecgh/go-spew/spew",
    ],
)
