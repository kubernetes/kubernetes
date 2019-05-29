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

CRI_TOOLS_VERSION = "1.14.0"
_CRI_TARBALL_ARCH_SHA256 = {
    "amd64": "483c90a9fe679590df4332ba807991c49232e8cd326c307c575ecef7fe22327b",
    "arm": "9910cecfd6558239ba015323066c7233d8371af359b9ddd0b2a35d5223bcf945",
    "arm64": "f76b3d00a272c8d210e9a45f77d07d3770bee310d99c4fd9a72d6f55278882e5",
    "ppc64le": "1e2cd11a1e025ed9755521cf13bb1bda986afa0052597a9bb44d31e62583413b",
    "s390x": "8b7b5749cba88ef337997ae90aa04380e3cab2c040b44b505b2fcd691c4935e4",
}

ETCD_VERSION = "3.3.10"
_ETCD_TARBALL_ARCH_SHA256 = {
    "amd64": "1620a59150ec0a0124a65540e23891243feb2d9a628092fb1edcc23974724a45",
    "arm64": "5ec97b0b872adce275b8130d19db314f7f2b803aeb24c4aae17a19e2d66853c4",
    "ppc64le": "148fe96f0ec1813c5db9916199e96a913174304546bc8447a2d2f9fee4b8f6c2",
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
            urls = mirror("https://github.com/kubernetes-incubator/cri-tools/releases/download/v%s/crictl-v%s-linux-%s.tar.gz" % (CRI_TOOLS_VERSION, CRI_TOOLS_VERSION, arch)),
        )

# Use go get -u github.com/estesp/manifest-tool to find these values
_DEBIAN_BASE_DIGEST = {
    "manifest": "sha256:6966a0aedd7592c18ff2dd803c08bd85780ee19f5e3a2e7cf908a4cd837afcde",
    "amd64": "sha256:8ccb65cd2dd7e0c24193d0742a20e4a673dbd11af5a33f16fcd471a31486866c",
    "arm": "sha256:3432b41de3f6dfffdc1386fce961cfd1f9f8e208b3a35070e10ef3e2a733cb17",
    "arm64": "sha256:9189251e1d1eb4126d6e6add2e272338f9c8a6a3db38863044625bca4b667f31",
    "ppc64le": "sha256:50aa659e1e75e4231ee8293c3b4115e5755bb0517142b9b4bddbc134bf4354db",
    "s390x": "sha256:bbb8ee3a2aaca738c00809f450233d98029fea4e319d8faaa30aa94c8b17a806",
}

_DEBIAN_IPTABLES_DIGEST = {
    "manifest": "sha256:b522b0035dba3ac2d5c0dbaaf8217bd66248e790332ccfdf653e0f943a280dcf",
    "amd64": "sha256:adc40e9ec817c15d35b26d1d6aa4d0f8096fba4c99e26a026159bb0bc98c6a89",
    "arm": "sha256:58e8a1d3b187eed2d8d3664cd1c9723e5029698714a24dfca4b6ef42ea27a9d4",
    "arm64": "sha256:1a63fdd216fe7b84561d40ab1ebaa0daae1fc73e4232a6caffbd8353d9a14cea",
    "ppc64le": "sha256:9f90adbc7513cc96d92fcec7633c4b29e766dd31cf876af03c0b54374e22fa9c",
    "s390x": "sha256:4f147708deff2a0163ee49b6980cc95423514bec5f4091612d65773b898fbdae",
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
            tag = "0.4.1",  # ignored, but kept here for documentation
        )

        container_pull(
            name = "debian-iptables-" + arch,
            architecture = arch,
            digest = _digest(_DEBIAN_IPTABLES_DIGEST, arch),
            registry = "k8s.gcr.io",
            repository = "debian-iptables",
            tag = "v11.0.2",  # ignored, but kept here for documentation
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
