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

ETCD_VERSION = "3.3.17"
_ETCD_TARBALL_ARCH_SHA256 = {
    "amd64": "8c1168a24d17a2d6772f8148ea35d4f3398c51f1e23db90c849d506adb387060",
    "arm64": "0ea20dfbf3085f584f788287fd398979d0f1271549be6497d81ec635b9b4c121",
    "ppc64le": "49cef090eb67f0d24ebd1733303ed2088c291ad47c60f57746da9f689aa1de7a",
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
