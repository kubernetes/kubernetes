package(default_visibility = ["//visibility:public"])

load("//build:platforms.bzl", "CLIENT_PLATFORMS", "NODE_PLATFORMS", "for_platforms")
load("//build:workspace.bzl", "CNI_VERSION", "CRI_TOOLS_VERSION")
load("@io_k8s_repo_infra//defs:build.bzl", "release_filegroup")
load("@io_k8s_repo_infra//defs:rpm.bzl", "pkg_rpm_for_goarch")

release_filegroup(
    name = "rpms",
    conditioned_srcs = for_platforms(
        default = [],
        for_client = [":kubectl.rpm"],
        for_node = [
            ":cri-tools.rpm",
            ":kubeadm.rpm",
            ":kubelet.rpm",
            ":kubernetes-cni.rpm",
        ],
        only_os = "linux",
    ),
    tags = ["manual"],
    visibility = ["//visibility:public"],
)

# Create genrules to copy the arch-specific RPMs to RPMs without the arch in their filename.
genrule(
    name = "kubectl",
    srcs = select(for_platforms(
        for_client = [":kubectl-{ARCH}.rpm"],
        only_os = "linux",
    )),
    outs = ["kubectl.rpm"],
    cmd = "cp $< $@",
    output_to_bindir = True,
)

[genrule(
    name = pkg,
    srcs = select(for_platforms(
        for_client = [":%s-{ARCH}.rpm" % pkg],
        only_os = "linux",
    )),
    outs = ["%s.rpm" % pkg],
    cmd = "cp $< $@",
    output_to_bindir = True,
) for pkg in [
    "cri-tools",
    "kubeadm",
    "kubelet",
    "kubernetes-cni",
]]

[pkg_rpm_for_goarch(
    name = "kubectl",
    changelog = "//:CHANGELOG.md",
    data = [
        "//cmd/kubectl",
    ],
    goarch = arch,
    spec_file = "kubectl.spec",
    tags = ["manual"],
    version_file = "//build:os_package_version",
) for arch in CLIENT_PLATFORMS["linux"]]

[pkg_rpm_for_goarch(
    name = "kubelet",
    changelog = "//:CHANGELOG.md",
    data = [
        "kubelet.service",
        "//cmd/kubelet",
    ],
    goarch = arch,
    spec_file = "kubelet.spec",
    tags = ["manual"],
    version_file = "//build:os_package_version",
) for arch in NODE_PLATFORMS["linux"]]

[pkg_rpm_for_goarch(
    name = "kubeadm",
    changelog = "//:CHANGELOG.md",
    data = [
        "10-kubeadm.conf",
        "50-kubeadm.conf",
        "kubeadm.conf",
        "kubelet.env",
        "//cmd/kubeadm",
    ],
    goarch = arch,
    spec_file = "kubeadm.spec",
    tags = ["manual"],
    version_file = "//build:os_package_version",
) for arch in NODE_PLATFORMS["linux"]]

[pkg_rpm_for_goarch(
    name = "kubernetes-cni",
    changelog = "//:CHANGELOG.md",
    data = [
        "@kubernetes_cni_{GOARCH}//file",
    ],
    goarch = arch,
    spec_file = "kubernetes-cni.spec",
    tags = ["manual"],
    version = CNI_VERSION,
) for arch in NODE_PLATFORMS["linux"]]

[pkg_rpm_for_goarch(
    name = "cri-tools",
    data = [
        "@cri_tools_linux-{GOARCH}//file",
    ],
    goarch = arch,
    spec_file = "cri-tools.spec",
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
