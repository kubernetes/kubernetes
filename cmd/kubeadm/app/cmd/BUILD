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
        "//cmd/kubeadm/app/apis/kubeadm/validation:go_default_library",
        "//cmd/kubeadm/app/cmd/flags:go_default_library",
        "//cmd/kubeadm/app/discovery:go_default_library",
        "//cmd/kubeadm/app/master:go_default_library",
        "//cmd/kubeadm/app/node:go_default_library",
        "//cmd/kubeadm/app/phases/certs:go_default_library",
        "//cmd/kubeadm/app/phases/kubeconfig:go_default_library",
        "//cmd/kubeadm/app/preflight:go_default_library",
        "//cmd/kubeadm/app/util:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/fields:go_default_library",
        "//pkg/kubectl:go_default_library",
        "//pkg/kubectl/cmd/util:go_default_library",
        "//pkg/util/flag:go_default_library",
        "//pkg/util/initsystem:go_default_library",
        "//pkg/version:go_default_library",
        "//vendor:github.com/renstrom/dedent",
        "//vendor:github.com/spf13/cobra",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "reset_test.go",
        "token_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = ["//cmd/kubeadm/app/preflight:go_default_library"],
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
        "//cmd/kubeadm/app/cmd/flags:all-srcs",
    ],
    tags = ["automanaged"],
)
