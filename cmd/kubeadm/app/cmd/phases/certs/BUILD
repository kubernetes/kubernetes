load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["renew.go"],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/cmd/phases/certs",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/kubeadm/app/apis/kubeadm/scheme:go_default_library",
        "//cmd/kubeadm/app/apis/kubeadm/v1alpha3:go_default_library",
        "//cmd/kubeadm/app/cmd/options:go_default_library",
        "//cmd/kubeadm/app/cmd/util:go_default_library",
        "//cmd/kubeadm/app/constants:go_default_library",
        "//cmd/kubeadm/app/phases/certs:go_default_library",
        "//cmd/kubeadm/app/phases/certs/renewal:go_default_library",
        "//cmd/kubeadm/app/util:go_default_library",
        "//cmd/kubeadm/app/util/config:go_default_library",
        "//cmd/kubeadm/app/util/kubeconfig:go_default_library",
        "//pkg/util/normalizer:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["renewal_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//cmd/kubeadm/app/constants:go_default_library",
        "//cmd/kubeadm/app/phases/certs/pkiutil:go_default_library",
        "//cmd/kubeadm/test:go_default_library",
        "//cmd/kubeadm/test/certs:go_default_library",
        "//cmd/kubeadm/test/cmd:go_default_library",
        "//vendor/github.com/spf13/cobra:go_default_library",
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
