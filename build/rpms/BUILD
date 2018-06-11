package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_defs/pkg:rpm.bzl", "pkg_rpm")
load("//build:workspace.bzl", "CRI_TOOLS_VERSION")

filegroup(
    name = "rpms",
    srcs = [
        ":cri-tools",
        ":kubeadm",
        ":kubectl",
        ":kubelet",
        ":kubernetes-cni",
    ],
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

pkg_rpm(
    name = "kubectl",
    architecture = "x86_64",
    changelog = "//:CHANGELOG.md",
    data = [
        "//cmd/kubectl",
    ],
    spec_file = "kubectl.spec",
    tags = ["manual"],
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
    tags = ["manual"],
    version_file = "//build:os_package_version",
)

pkg_rpm(
    name = "kubeadm",
    architecture = "x86_64",
    changelog = "//:CHANGELOG.md",
    data = [
        "10-kubeadm.conf",
        "kubelet.env",
        "//cmd/kubeadm",
    ],
    spec_file = "kubeadm.spec",
    tags = ["manual"],
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
    tags = ["manual"],
    version_file = "//build:cni_package_version",
)

pkg_rpm(
    name = "cri-tools",
    architecture = "x86_64",
    data = [
        "@cri_tools//file",
    ],
    spec_file = "cri-tools.spec",
    tags = ["manual"],
    # dashes are not allowed in rpm versions
    version = CRI_TOOLS_VERSION.replace("-", "_"),
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
