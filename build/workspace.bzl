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

CNI_VERSION = "0.8.5"
_CNI_TARBALL_ARCH_SHA256 = {
    "amd64": "bd682ffcf701e8f83283cdff7281aad0c83b02a56084d6e601216210732833f9",
    "arm": "86a868234045837cb3f5d58a0a4468ff42845d50b5e87bd128f050ef393d7495",
    "arm64": "a7881ec37e592c897bdfd2a225b4ed74caa981e3c4cdcf8f45574f8d2f111bce",
    "ppc64le": "a26cc3734f7cb980ab8fb3aaa64ccf2d67291478130009fa9542355fbdd94aa5",
    "s390x": "033ea910a83144609083d5c3fb62bf4a379b0b17729a1a9e829feed9fa7a9d97",
}

CRI_TOOLS_VERSION = "1.17.0"
_CRI_TARBALL_ARCH_SHA256 = {
    "linux-386": "cffa443cf76ab4b760a68d4db555d1854cb692e8b20b3360cf23221815ca151e",
    "linux-amd64": "7b72073797f638f099ed19550d52e9b9067672523fc51b746e65d7aa0bafa414",
    "linux-arm": "9700957218e8e7bdc02cbc8fda4c189f5b6223a93ba89d876bdfd77b6117e9b7",
    "linux-arm64": "d89afd89c2852509fafeaff6534d456272360fcee732a8d0cb89476377387e12",
    "linux-ppc64le": "a61c52b9ac5bffe94ae4c09763083c60f3eccd30eb351017b310f32d1cafb855",
    "linux-s390x": "0db445f0b74ecb51708b710480a462b728174155c5f2709a39d1cc2dc975e350",
    "windows-386": "2e285250d36b5cb3e8c047b191c0c0af606fed7c0034bb140ba95cc1498f4996",
    "windows-amd64": "e18150d5546d3ddf6b165bd9aec0f65c18aacf75b94fb28bb26bfc0238f07b28",
}

ETCD_VERSION = "3.4.3"
_ETCD_TARBALL_ARCH_SHA256 = {
    "amd64": "6c642b723a86941b99753dff6c00b26d3b033209b15ee33325dc8e7f4cd68f07",
    "arm64": "01bd849ad99693600bd59db8d0e66ac64aac1e3801900665c31bd393972e3554",
    "ppc64le": "3f20888d6efb7f2665ebe278860eec6e8fc9555624e56c3d93f5a6b6dd90a21a",
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
            urls = mirror("https://github.com/kubernetes-incubator/cri-tools/releases/download/v%s/crictl-v%s-%s.tar.gz" % (CRI_TOOLS_VERSION, CRI_TOOLS_VERSION, arch)),
        )

# Use go get -u github.com/estesp/manifest-tool to find these values
_DEBIAN_BASE_DIGEST = {
    "manifest": "sha256:ebda8587ec0f49eb88ee3a608ef018484908cbc5aa32556a0d78356088c185d4",
    "amd64": "sha256:d7be39e143d4e6677a28c81c0a84868b40800fc979dea1848bb19d526668a00c",
    "arm": "sha256:fc731da13b0bc9013b85a86b583fc92e50869b5bc8e7aa6ca730ec0240954c7d",
    "arm64": "sha256:12502c3eed050fa9b6d5fe353a44bfc5f437dc325c8912b1a48dcc180df36f1e",
    "ppc64le": "sha256:4277aa59b63c5a1369e6d84a295ecc4ffa08985dcf114de9f7b6de1af4fcbc86",
    "s390x": "sha256:78ef2a6b017539379c1654b4e52ba8519bfec821c62d0b3a1dbd15104b711e21",
}

_DEBIAN_IPTABLES_DIGEST = {
    "manifest": "sha256:d1cd487e89fb4cba853cd3a948a6e9016faf66f2a7bb53cb1ac6b6c9cb58f5ed",
    "amd64": "sha256:852d3c569932059bcab3a52cb6105c432d85b4b7bbd5fc93153b78010e34a783",
    "arm": "sha256:c10f01b414a7cd4b2f3e26e152c90c64a1e781d99f83a6809764cf74ecbc46c3",
    "arm64": "sha256:5725e6fde13a6405cf800e22846ebd2bde24b0860f1dc3f6f5f256f03cfa85bd",
    "ppc64le": "sha256:b6d6e56a0c34c0393dcba0d5faaa531b92e5876114c5ab5a90e82e4889724c5a",
    "s390x": "sha256:39e67e9bf25d67fe35bd9dcb25367277e5967368e02f2741e0efd4ce8874db14",
}

_DEBIAN_HYPERKUBE_BASE_DIGEST = {
    "manifest": "sha256:8cabe02be6e86685d8860b7ace7c7addc9591a339728703027a4854677f1c772",
    "amd64": "sha256:5d4ea2fb5fbe9a9a9da74f67cf2faefc881968bc39f2ac5d62d9167e575812a1",
    "arm": "sha256:73260814af61522ff6aa48291df457d3bb0a91c4bf72e7cfa51fbaf03eb65fae",
    "arm64": "sha256:78eeb1a31eef7c16f954444d64636d939d89307e752964ad6d9d06966c722da3",
    "ppc64le": "sha256:92857d647abe8d9c7b4d7160cd5699112afc12fde369082a8ed00688b17928a9",
    "s390x": "sha256:c11d74fa0538c67238576c247bfaddf95ebaa90cd03cb4d2f2ac3c6ebe0441e2",
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
            registry = "k8s.gcr.io",
            repository = "debian-base",
            tag = "v2.0.0",  # ignored, but kept here for documentation
        )

        container_pull(
            name = "debian-iptables-" + arch,
            architecture = arch,
            digest = _digest(_DEBIAN_IPTABLES_DIGEST, arch),
            registry = "k8s.gcr.io",
            repository = "debian-iptables",
            tag = "v12.0.1",  # ignored, but kept here for documentation
        )

        container_pull(
            name = "debian-hyperkube-base-" + arch,
            architecture = arch,
            digest = _digest(_DEBIAN_HYPERKUBE_BASE_DIGEST, arch),
            registry = "k8s.gcr.io",
            repository = "debian-hyperkube-base",
            tag = "0.12.1",  # ignored, but kept here for documentation
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
