load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "certs.go",
        "constant.go",
        "doc.go",
        "generic.go",
        "token.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/cmd/options",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm/v1beta1:go_default_library",
        "//cmd/kubeadm/app/constants:go_default_library",
        "//staging/src/k8s.io/cluster-bootstrap/token/api:go_default_library",
        "//staging/src/k8s.io/component-base/cli/flag:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
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
    visibility = ["//visibility:public"],
)
