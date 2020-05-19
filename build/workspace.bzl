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

CNI_VERSION = "0.7.5"
_CNI_TARBALL_ARCH_SHA256 = {
    "amd64": "3ca15c0a18ee830520cf3a95408be826cbd255a1535a38e0be9608b25ad8bf64",
    "arm": "0eb4a528b5b2e4ce23ebc96e41b2f5280d5a64d41eec8dd8b16c3d66aaa0f6b8",
    "arm64": "7fec91af78e9548df306f0ec43bea527c8c10cc3a9682c33e971c8522a7fcded",
    "ppc64le": "9164a26ed8dd398b2fe3b15d9d456271dfa59aa537528d10572ea9fa2cef7679",
    "s390x": "415cdcf02c65c22f5b7e55b0ab61208a10f2b95a0c8310176c771d07a9f448cf",
}

CRI_TOOLS_VERSION = "1.16.1"
_CRI_TARBALL_ARCH_SHA256 = {
    "linux-386": "35b721a7a90a12cf9689fb1f4fe5b50f73520200130a52b552234bd798d2ae9f",
    "linux-amd64": "19fed421710fccfe58f5573383bb137c19438a9056355556f1a15da8d23b3ad1",
    "linux-arm": "367826f3eb06c4d923f3174d23141ddacef9ffcb0c902502bd922dbad86d08dd",
    "linux-arm64": "62b60ab7046b788df892a1b746bd602c520a59c38232febc0580692c9805f641",
    "linux-ppc64le": "d6464188a5011242b8dad22cb1e55d8cb29d6873f3d1f3f3c32bb236d3fca64e",
    "linux-s390x": "f3d5e707810d7985f6a470ea439ca5989c0ee218a824795ed1726f4958281a2d",
    "windows-386": "b7564675f69aa2a01b092874c9c9aeda084204384d711a6fb85c6d2328ca5c7d",
    "windows-amd64": "7d092dcb3b1af2edf75477d5d049a70e8c0d1ac8242b1dff2de7e6aa084e3615",
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
            urls = mirror("https://storage.googleapis.com/kubernetes-release/network-plugins/cni-plugins-%s-v%s.tgz" % (arch, CNI_VERSION)),
        )

def cri_tarballs():
    for arch, sha in _CRI_TARBALL_ARCH_SHA256.items():
        http_file(
            name = "cri_tools_%s" % arch,
            downloaded_file_path = "cri_tools.tgz",
            sha256 = sha,
            urls = mirror("https://github.com/kubernetes-incubator/cri-tools/releases/download/v%s/crictl-v%s-%s.tar.gz" % (CRI_TOOLS_VERSION, CRI_TOOLS_VERSION, arch)),
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
