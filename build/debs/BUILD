package(default_visibility = ["//visibility:public"])

load("@io_k8s_repo_infra//defs:deb.bzl", "deb_data", "k8s_deb")
load("@io_k8s_repo_infra//defs:build.bzl", "release_filegroup")
load("@io_k8s_repo_infra//defs:pkg.bzl", "pkg_tar")
load(
    "//build:platforms.bzl",
    "CLIENT_PLATFORMS",
    "NODE_PLATFORMS",
    "SERVER_PLATFORMS",
    "for_platforms",
)
load("//build:workspace.bzl", "CNI_VERSION", "CRI_TOOLS_VERSION")

# We do not include kube-scheduler, kube-controller-manager,
# kube-apiserver, and kube-proxy in this list even though we
# produce debs for them. We recommend that they be run in docker
# images. We use the debs that we produce here to build those
# images.
release_filegroup(
    name = "debs",
    conditioned_srcs = for_platforms(
        default = [],
        for_client = [":kubectl.deb"],
        for_node = [
            ":cri-tools.deb",
            ":kubeadm.deb",
            ":kubelet.deb",
            ":kubernetes-cni.deb",
        ],
        only_os = "linux",
    ),
)

# Create genrules to copy the arch-specific debs to debs without the arch in their filename.
genrule(
    name = "kubectl",
    srcs = select(for_platforms(
        for_client = [":kubectl-{ARCH}.deb"],
        only_os = "linux",
    )),
    outs = ["kubectl.deb"],
    cmd = "cp $< $@",
    output_to_bindir = True,
)

[genrule(
    name = pkg,
    srcs = select(for_platforms(
        for_node = [":%s-{ARCH}.deb" % pkg],
        only_os = "linux",
    )),
    outs = ["%s.deb" % pkg],
    cmd = "cp $< $@",
    output_to_bindir = True,
) for pkg in [
    "cri-tools",
    "kubeadm",
    "kubelet",
    "kubernetes-cni",
]]

[deb_data(
    name = "kubectl",
    data = [
        {
            "files": ["//cmd/kubectl"],
            "mode": "0755",
            "dir": "/usr/bin",
        },
    ],
    goarch = arch,
    tags = ["manual"],
) for arch in CLIENT_PLATFORMS["linux"]]

[[deb_data(
    name = binary,
    data = [
        {
            "files": ["//cmd/" + binary],
            "mode": "0755",
            "dir": "/usr/bin",
        },
    ],
    goarch = arch,
    tags = ["manual"],
) for binary in [
    "cloud-controller-manager",
    "kube-apiserver",
    "kube-controller-manager",
    "kube-proxy",
    "kube-scheduler",
]] for arch in SERVER_PLATFORMS["linux"]]

[deb_data(
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
    goarch = arch,
    tags = ["manual"],
) for arch in NODE_PLATFORMS["linux"]]

[deb_data(
    name = "kubeadm",
    data = [
        {
            "files": ["//cmd/kubeadm"],
            "mode": "0755",
            "dir": "/usr/bin",
        },
        {
            "files": ["10-kubeadm.conf"],
            "mode": "644",
            "dir": "/etc/systemd/system/kubelet.service.d",
        },
        {
            "files": ["kubeadm.conf"],
            "mode": "644",
            "dir": "/usr/lib/modules-load.d",
        },
        {
            "files": ["50-kubeadm.conf"],
            "mode": "644",
            "dir": "/etc/sysctl.d",
        },
    ],
    goarch = arch,
    tags = ["manual"],
) for arch in NODE_PLATFORMS["linux"]]

[pkg_tar(
    name = "kubernetes-cni-data-%s" % goarch,
    package_dir = "/opt/cni/bin",
    tags = ["manual"],
    deps = ["@kubernetes_cni_%s//file" % goarch],
) for goarch in NODE_PLATFORMS["linux"]]

[pkg_tar(
    name = "cri-tools-data-%s" % goarch,
    package_dir = "/usr/bin",
    tags = ["manual"],
    deps = ["@cri_tools_%s//file" % goarch],
) for goarch in NODE_PLATFORMS["linux"]]

[k8s_deb(
    name = "cloud-controller-manager",
    description = "Kubernetes Cloud Controller Manager",
    goarch = arch,
    tags = ["manual"],
    version_file = "//build:os_package_version",
) for arch in SERVER_PLATFORMS["linux"]]

[k8s_deb(
    name = "kubectl",
    description = """Kubernetes Command Line Tool
The Kubernetes command line tool for interacting with the Kubernetes API.
""",
    goarch = arch,
    tags = ["manual"],
    version_file = "//build:os_package_version",
) for arch in CLIENT_PLATFORMS["linux"]]

[k8s_deb(
    name = "kube-apiserver",
    description = "Kubernetes API Server",
    goarch = arch,
    tags = ["manual"],
    version_file = "//build:os_package_version",
) for arch in SERVER_PLATFORMS["linux"]]

[k8s_deb(
    name = "kube-controller-manager",
    description = "Kubernetes Controller Manager",
    goarch = arch,
    tags = ["manual"],
    version_file = "//build:os_package_version",
) for arch in SERVER_PLATFORMS["linux"]]

[k8s_deb(
    name = "kube-scheduler",
    description = "Kubernetes Scheduler",
    goarch = arch,
    tags = ["manual"],
    version_file = "//build:os_package_version",
) for arch in SERVER_PLATFORMS["linux"]]

[k8s_deb(
    name = "kube-proxy",
    depends = [
        "iptables (>= 1.4.21)",
        "iproute2",
    ],
    description = "Kubernetes Service Proxy",
    goarch = arch,
    tags = ["manual"],
    version_file = "//build:os_package_version",
) for arch in NODE_PLATFORMS["linux"]]

[k8s_deb(
    name = "kubelet",
    depends = [
        "conntrack",
        "ebtables",
        "ethtool",
        "iproute2",
        "iptables (>= 1.4.21)",
        "kubernetes-cni (>= %s)" % CNI_VERSION,
        "mount",
        "socat",
        "util-linux",
    ],
    description = """Kubernetes Node Agent
The node agent of Kubernetes, the container cluster manager
""",
    goarch = arch,
    tags = ["manual"],
    version_file = "//build:os_package_version",
) for arch in NODE_PLATFORMS["linux"]]

[k8s_deb(
    name = "kubeadm",
    depends = [
        "kubelet (>= 1.8.0)",
        "kubectl (>= 1.8.0)",
        "kubernetes-cni (>= %s)" % CNI_VERSION,
        "cri-tools (>= 1.11.0)",
    ],
    description = """Kubernetes Cluster Bootstrapping Tool
The Kubernetes command line tool for bootstrapping a Kubernetes cluster.
""",
    goarch = arch,
    postinst = "postinst",
    tags = ["manual"],
    version_file = "//build:os_package_version",
) for arch in NODE_PLATFORMS["linux"]]

[k8s_deb(
    name = "kubernetes-cni",
    description = """Kubernetes Packaging of CNI
The Container Networking Interface tools for provisioning container networks.
""",
    goarch = arch,
    tags = ["manual"],
    version = CNI_VERSION,
) for arch in NODE_PLATFORMS["linux"]]

[k8s_deb(
    name = "cri-tools",
    description = """Container Runtime Interface tools (crictl)""",
    goarch = arch,
    tags = ["manual"],
    version = CRI_TOOLS_VERSION,
) for arch in NODE_PLATFORMS["linux"]]

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
