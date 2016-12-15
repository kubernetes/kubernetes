package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar", "pkg_deb")
load("@io_kubernetes_build//defs:deb.bzl", "k8s_deb", "deb_data")

filegroup(
    name = "debs",
    srcs = [
        ":kubeadm.deb",
        ":kubectl.deb",
        ":kubelet.deb",
        ":kubernetes-cni.deb",
    ],
)

deb_data(
    name = "kubectl",
    data = [
        {
            "files": ["//cmd/kubectl"],
            "mode": "0755",
            "dir": "/usr/bin",
        },
    ],
)

deb_data(
    name = "kubelet",
    data = [
        {
            "files": ["//cmd/kubelet"],
            "mode": "0755",
            "dir": "/usr/bin",
        },
        {
            "files": ["kubelet.service"],
            "mode": "644",
            "dir": "/lib/systemd/system",
        },
    ],
)

deb_data(
    name = "kubeadm",
    data = [
        {
            "files": ["//cmd/kubeadm"],
            "mode": "0755",
            "dir": "/usr/bin",
        },
        {
            "files": ["kubeadm-10.conf"],
            "mode": "644",
            "dir": "/etc/systemd/system/kubelet.service.d",
        },
    ],
)

pkg_tar(
    name = "kubernetes-cni-data",
    package_dir = "/opt/cni",
    deps = ["@kubernetes_cni//file"],
)

k8s_deb(
    name = "kubectl",
    description = """Kubernetes Command Line Tool
The Kubernetes command line tool for interacting with the Kubernetes API.
""",
)

k8s_deb(
    name = "kubelet",
    depends = [
        "iptables (>= 1.4.21)",
        "kubernetes-cni (>= 0.3.0.1)",
        "iproute2",
        "socat",
        "util-linux",
        "mount",
        "ebtables",
        "ethtool",
    ],
    description = """Kubernetes Node Agent
The node agent of Kubernetes, the container cluster manager
""",
)

k8s_deb(
    name = "kubeadm",
    depends = [
        "kubelet (>= 1.4.0)",
        "kubectl (>= 1.4.0)",
    ],
    description = """Kubernetes Cluster Bootstrapping Tool
The Kubernetes command line tool for bootstrapping a Kubernetes cluster.
""",
)

k8s_deb(
    name = "kubernetes-cni",
    description = """Kubernetes Packaging of CNI
The Container Networking Interface tools for provisioning container networks.
""",
)
