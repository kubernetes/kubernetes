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
        "cmd.go",
        "init.go",
        "join.go",
        "reset.go",
        "token.go",
        "version.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm:go_default_library",
        "//cmd/kubeadm/app/apis/kubeadm/v1alpha1:go_default_library",
        "//cmd/kubeadm/app/master:go_default_library",
        "//cmd/kubeadm/app/node:go_default_library",
        "//cmd/kubeadm/app/preflight:go_default_library",
        "//cmd/kubeadm/app/util:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/cloudprovider/providers:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/flag:go_default_library",
        "//pkg/util/initsystem:go_default_library",
        "//pkg/util/net:go_default_library",
        "//pkg/version:go_default_library",
        "//vendor:github.com/renstrom/dedent",
        "//vendor:github.com/spf13/cobra",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "reset_test.go",
        "token_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = ["//cmd/kubeadm/app/preflight:go_default_library"],
)
