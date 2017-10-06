package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["kubeadm.go"],
    deps = [
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/kubeadm/app/apis/kubeadm/install:go_default_library",
        "//vendor/k8s.io/kubeadm/app/cmd:go_default_library",
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
        "//staging/src/k8s.io/kubeadm/app/apis/kubeadm:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/cmd:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/constants:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/discovery:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/features:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/images:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/node:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/phases/addons/dns:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/phases/addons/proxy:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/phases/bootstraptoken/clusterinfo:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/phases/bootstraptoken/node:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/phases/certs:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/phases/controlplane:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/phases/etcd:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/phases/kubeconfig:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/phases/markmaster:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/phases/selfhosting:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/phases/token:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/phases/upgrade:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/phases/uploadconfig:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/preflight:all-srcs",
        "//staging/src/k8s.io/kubeadm/app/util:all-srcs",
    ],
    tags = ["automanaged"],
)
