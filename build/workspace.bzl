# Copyright 2018 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("//build:platforms.bzl", "SERVER_PLATFORMS")
load("//build:workspace_mirror.bzl", "mirror")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("@io_bazel_rules_docker//container:container.bzl", "container_pull")

CNI_VERSION = "0.8.6"
_CNI_TARBALL_ARCH_SHA256 = {
    "amd64": "994fbfcdbb2eedcfa87e48d8edb9bb365f4e2747a7e47658482556c12fd9b2f5",
    "arm": "28e61b5847265135dc1ca397bf94322ecce4acab5c79cc7d360ca3f6a655bdb7",
    "arm64": "43fbf750c5eccb10accffeeb092693c32b236fb25d919cf058c91a677822c999",
    "ppc64le": "61d6c6c15d3e4fa3eb85d128c9c0ff2658f38e59047ae359be47d193c673e116",
    "s390x": "ca126a3bd2cd8dff1c7bbfc3c69933b284c4e77614391c7e1f74b0851fc3b289",
}

CRI_TOOLS_VERSION = "1.14.0"
_CRI_TARBALL_ARCH_SHA256 = {
    "amd64": "483c90a9fe679590df4332ba807991c49232e8cd326c307c575ecef7fe22327b",
    "arm": "9910cecfd6558239ba015323066c7233d8371af359b9ddd0b2a35d5223bcf945",
    "arm64": "f76b3d00a272c8d210e9a45f77d07d3770bee310d99c4fd9a72d6f55278882e5",
    "ppc64le": "1e2cd11a1e025ed9755521cf13bb1bda986afa0052597a9bb44d31e62583413b",
    "s390x": "8b7b5749cba88ef337997ae90aa04380e3cab2c040b44b505b2fcd691c4935e4",
}

ETCD_VERSION = "3.3.15"
_ETCD_TARBALL_ARCH_SHA256 = {
    "amd64": "87e30dc472a48b775a9e764cb742a5df0a2844e7b82d12917d7ea489febbc8c8",
    "arm64": "0cdacf1a5f8095322ee77b62ac5e79c3326a5166deb69f558214b110b312b4c0",
    "ppc64le": "dcaa8d0ec1a117f6456778f3b9c7e685d4609c2a3c448afda1aa7a4232d0c481",
}

# Dependencies needed for a Kubernetes "release", e.g. building docker images,
# debs, RPMs, or tarballs.
def release_dependencies():
    cni_tarballs()
    cri_tarballs()
    debian_image_dependencies()
    etcd_tarballs()

def cni_tarballs():
    for arch, sha in _CNI_TARBALL_ARCH_SHA256.items():
        http_file(
            name = "kubernetes_cni_%s" % arch,
            downloaded_file_path = "kubernetes_cni.tgz",
            sha256 = sha,
            urls = ["https://storage.googleapis.com/k8s-artifacts-cni/release/v%s/cni-plugins-linux-%s-v%s.tgz" % (CNI_VERSION, arch, CNI_VERSION)],
        )

def cri_tarballs():
    for arch, sha in _CRI_TARBALL_ARCH_SHA256.items():
        http_file(
            name = "cri_tools_%s" % arch,
            downloaded_file_path = "cri_tools.tgz",
            sha256 = sha,
            urls = mirror("https://github.com/kubernetes-incubator/cri-tools/releases/download/v%s/crictl-v%s-linux-%s.tar.gz" % (CRI_TOOLS_VERSION, CRI_TOOLS_VERSION, arch)),
        )

# Use skopeo to find these values: https://github.com/containers/skopeo
#
# Example
# Manifest: skopeo inspect docker://gcr.io/k8s-staging-build-image/debian-base:v2.1.0
# Arches: skopeo inspect --raw docker://gcr.io/k8s-staging-build-image/debian-base:v2.1.0
_DEBIAN_BASE_DIGEST = {
    "manifest": "sha256:b118abac0bcf633b9db4086584ee718526fe394cf1bd18aee036e6cc497860f6",
    "amd64": "sha256:a67798e4746faaab3fde5b7407fa8bba75d8b1214d168dc7ad2b5364f6fc4319",
    "arm": "sha256:3ab4332e481610acbcba7a801711e29506b4bd4ecb38f72590253674d914c449",
    "arm64": "sha256:8d53ac4da977eb20d6219ee49b9cdff8c066831ecab0e4294d0a02179d26b1d7",
    "ppc64le": "sha256:a631023e795fe18df7faa8fe1264e757a6c74a232b9a2659657bf65756f3f4aa",
    "s390x": "sha256:dac908eaa61d2034aec252576a470a7e4ab184c361f89170526f707a0c3c6082",
}

# Use skopeo to find these values: https://github.com/containers/skopeo
#
# Example
# Manifest: skopeo inspect docker://gcr.io/k8s-staging-build-image/debian-iptables:v12.1.0
# Arches: skopeo inspect --raw docker://gcr.io/k8s-staging-build-image/debian-iptables:v12.1.0
_DEBIAN_IPTABLES_DIGEST = {
    "manifest": "sha256:1ae6d76dea462973759ff1c4e02263867da1f85db9aa10462a030ca421cbf0e9",
    "amd64": "sha256:2fb9fa09123a41e6369cac04eb29e26237fe9e43da8e18f676d18d8fffb906fc",
    "arm": "sha256:a0e97386c073a2990265938fa15dc0db575efdb4d13c0ea63a79e0590813a998",
    "arm64": "sha256:2a7df97e2c702d9852cc6234aff89b4671cd5b09086ac2b5383411315e5f115d",
    "ppc64le": "sha256:f5289a6494328b7ccb695e3add65b33ca380b77fcfc9715e474f0efe26e1c506",
    "s390x": "sha256:1b91a2788750552913377bf1bc99a095544dfb523d80a55674003c974c8e0905",
}

# Use skopeo to find these values: https://github.com/containers/skopeo
#
# Example
# Manifest: skopeo inspect docker://gcr.io/k8s-staging-build-image/debian-hyperkube-base:v1.0.0
# Arches: skopeo inspect --raw docker://gcr.io/k8s-staging-build-image/debian-hyperkube-base:v1.0.0
_DEBIAN_HYPERKUBE_BASE_DIGEST = {
    "manifest": "sha256:8c8d854d868fb08352f73dda94f9e0b998c7318b48ddc587a355d0cbaf687f14",
    "amd64": "sha256:73a8cb2bfd6707c8ed70c252e97bdccad8bc265a9a585db12b8e3dac50d6cd2a",
    "arm": "sha256:aee7c2958c6e0de896995e8b04c8173fd0a7bbb16cddb1f4668e3ae010b8c786",
    "arm64": "sha256:a74fc6d690e5c5e393fd509f50d7203fa5cd19bfbb127d4a3a996fd1ebcf35c4",
    "ppc64le": "sha256:54afd9a85d6ecbe0792496f36012e7902601fb8084347cdc195c8b0561da39a3",
    "s390x": "sha256:405a94e6b82f2eb89c773e0e9f6105eb73cde5605d4508ae36a150d922fe1c66",
}

def _digest(d, arch):
    if arch not in d:
        print("WARNING: %s not found in %r" % (arch, d))
        return d["manifest"]
    return d[arch]

def debian_image_dependencies():
    for arch in SERVER_PLATFORMS["linux"]:
        container_pull(
            name = "debian-base-" + arch,
            architecture = arch,
            digest = _digest(_DEBIAN_BASE_DIGEST, arch),
            registry = "us.gcr.io/k8s-artifacts-prod/build-image",
            repository = "debian-base",
            # Ensure the digests above are updated to match a new tag
            tag = "v2.1.0",  # ignored, but kept here for documentation
        )

        container_pull(
            name = "debian-iptables-" + arch,
            architecture = arch,
            digest = _digest(_DEBIAN_IPTABLES_DIGEST, arch),
            registry = "us.gcr.io/k8s-artifacts-prod/build-image",
            repository = "debian-iptables",
            # Ensure the digests above are updated to match a new tag
            tag = "v12.1.0",  # ignored, but kept here for documentation
        )

        container_pull(
            name = "debian-hyperkube-base-" + arch,
            architecture = arch,
            digest = _digest(_DEBIAN_HYPERKUBE_BASE_DIGEST, arch),
            registry = "us.gcr.io/k8s-artifacts-prod/build-image",
            repository = "debian-hyperkube-base",
            # Ensure the digests above are updated to match a new tag
            tag = "v1.0.0",  # ignored, but kept here for documentation
        )

def etcd_tarballs():
    for arch, sha in _ETCD_TARBALL_ARCH_SHA256.items():
        http_archive(
            name = "com_coreos_etcd_%s" % arch,
            build_file = "@//third_party:etcd.BUILD",
            sha256 = sha,
            strip_prefix = "etcd-v%s-linux-%s" % (ETCD_VERSION, arch),
            urls = mirror("https://github.com/coreos/etcd/releases/download/v%s/etcd-v%s-linux-%s.tar.gz" % (ETCD_VERSION, ETCD_VERSION, arch)),
        )
