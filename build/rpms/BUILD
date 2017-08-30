package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_defs/pkg:rpm.bzl", "pkg_rpm")

pkg_rpm(
    name = "kubectl",
    architecture = "x86_64",
    changelog = "//:CHANGELOG.md",
    data = [
        "//cmd/kubectl",
    ],
    spec_file = "kubectl.spec",
    version_file = "//build:os_package_version",
)

pkg_rpm(
    name = "kubelet",
    architecture = "x86_64",
    changelog = "//:CHANGELOG.md",
    data = [
        "kubelet.service",
        "//cmd/kubelet",
    ],
    spec_file = "kubelet.spec",
    version_file = "//build:os_package_version",
)

pkg_rpm(
    name = "kubeadm",
    architecture = "x86_64",
    changelog = "//:CHANGELOG.md",
    data = [
        "10-kubeadm.conf",
        "//cmd/kubeadm",
    ],
    spec_file = "kubeadm.spec",
    version_file = "//build:os_package_version",
)

pkg_rpm(
    name = "kubernetes-cni",
    architecture = "x86_64",
    changelog = "//:CHANGELOG.md",
    data = [
        "@kubernetes_cni//file",
    ],
    spec_file = "kubernetes-cni.spec",
    version_file = "//build:os_package_version",
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
)
