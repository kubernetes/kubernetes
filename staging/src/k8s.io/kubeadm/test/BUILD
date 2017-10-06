package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["util.go"],
    deps = [
        "//vendor/github.com/renstrom/dedent:go_default_library",
        "//vendor/k8s.io/kubeadm/app/apis/kubeadm:go_default_library",
        "//vendor/k8s.io/kubeadm/app/constants:go_default_library",
        "//vendor/k8s.io/kubeadm/app/phases/certs/pkiutil:go_default_library",
        "//vendor/k8s.io/kubeadm/test/certs:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//staging/src/k8s.io/kubeadm/test/certs:all-srcs",
        "//staging/src/k8s.io/kubeadm/test/cmd:all-srcs",
        "//staging/src/k8s.io/kubeadm/test/kubeconfig:all-srcs",
    ],
    tags = ["automanaged"],
)
