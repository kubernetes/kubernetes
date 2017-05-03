package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_defs/pkg:rpm.bzl", "pkg_rpm")

pkg_rpm(
    name = "kubectl",
    changelog = "CHANGELOG",
    data = [
        "//cmd/kubectl",
    ],
    spec_file = "kubectl.spec",
    version = "1.6.0",
)

pkg_rpm(
    name = "kubelet",
    changelog = "CHANGELOG",
    data = [
        "kubelet.service",
        "//cmd/kubelet",
    ],
    spec_file = "kubelet.spec",
    version = "1.6.0",
)

pkg_rpm(
    name = "kubeadm",
    changelog = "CHANGELOG",
    data = [
        "10-kubeadm.conf",
        "//cmd/kubeadm",
    ],
    spec_file = "kubeadm.spec",
    version = "1.6.0",
)

pkg_rpm(
    name = "kubernetes-cni",
    changelog = "CHANGELOG",
    data = [
        "@kubernetes_cni//file",
    ],
    spec_file = "kubernetes-cni.spec",
    version = "1.6.0",
)
