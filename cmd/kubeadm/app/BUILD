package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["kubeadm.go"],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app",
    deps = [
        "//cmd/kubeadm/app/cmd:go_default_library",
        "//staging/src/k8s.io/component-base/cli/flag:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
        "//cmd/kubeadm/app/apis/kubeadm:all-srcs",
        "//cmd/kubeadm/app/apis/output:all-srcs",
        "//cmd/kubeadm/app/cmd:all-srcs",
        "//cmd/kubeadm/app/componentconfigs:all-srcs",
        "//cmd/kubeadm/app/constants:all-srcs",
        "//cmd/kubeadm/app/discovery:all-srcs",
        "//cmd/kubeadm/app/features:all-srcs",
        "//cmd/kubeadm/app/images:all-srcs",
        "//cmd/kubeadm/app/phases/addons/dns:all-srcs",
        "//cmd/kubeadm/app/phases/addons/proxy:all-srcs",
        "//cmd/kubeadm/app/phases/bootstraptoken/clusterinfo:all-srcs",
        "//cmd/kubeadm/app/phases/bootstraptoken/node:all-srcs",
        "//cmd/kubeadm/app/phases/certs:all-srcs",
        "//cmd/kubeadm/app/phases/controlplane:all-srcs",
        "//cmd/kubeadm/app/phases/copycerts:all-srcs",
        "//cmd/kubeadm/app/phases/etcd:all-srcs",
        "//cmd/kubeadm/app/phases/kubeconfig:all-srcs",
        "//cmd/kubeadm/app/phases/kubelet:all-srcs",
        "//cmd/kubeadm/app/phases/markcontrolplane:all-srcs",
        "//cmd/kubeadm/app/phases/patchnode:all-srcs",
        "//cmd/kubeadm/app/phases/selfhosting:all-srcs",
        "//cmd/kubeadm/app/phases/upgrade:all-srcs",
        "//cmd/kubeadm/app/phases/uploadconfig:all-srcs",
        "//cmd/kubeadm/app/preflight:all-srcs",
        "//cmd/kubeadm/app/util:all-srcs",
    ],
    tags = ["automanaged"],
)
