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

CNI_VERSION = "0.8.7"
_CNI_TARBALL_ARCH_SHA256 = {
    "amd64": "977824932d5667c7a37aa6a3cbba40100a6873e7bd97e83e8be837e3e7afd0a8",
    "arm": "5757778f4c322ffd93d7586c60037b81a2eb79271af6f4edf9ff62b4f7868ed9",
    "arm64": "ae13d7b5c05bd180ea9b5b68f44bdaa7bfb41034a2ef1d68fd8e1259797d642f",
    "ppc64le": "70a8c5448ed03a3b27c6a89499a05285760a45252ec7eae4190c70ba5400d4ba",
    "s390x": "3a0008f98ea5b4b6fd367cac3d8096f19bc080a779cf81fd0bcbc5bd1396ace7",
}

CRI_TOOLS_VERSION = "1.19.0"
_CRI_TARBALL_ARCH_SHA256 = {
    "linux-386": "fd0247b81a46adeca69ef3b7bbcf7d0e776df63195918236887243773b98a0c0",
    "linux-amd64": "87d8ef70b61f2fe3d8b4a48f6f712fd798c6e293ed3723c1e4bbb5052098f0ae",
    "linux-arm": "b72fd3c4b35f60f5db2cfcd8e932f6000cf9c2978b54adfcf60ee5e2d452e92f",
    "linux-arm64": "ec040d14ca03e8e4e504a85dae5353e04b5d9d8aea3df68699258992c0eb8d88",
    "linux-ppc64le": "72107c58960ee9405829c3366dbfcd86f163a990ea2102f3ed63a709096bc7ba",
    "linux-s390x": "20ec106c307c9d56c2ecae1560b244f8ac26450b9704682f24bfb5f468b06776",
    "windows-386": "3b7a41b556e3eae1fb56d17edc990ccd4839c8ab554249a8991155ee266dac4b",
    "windows-amd64": "df60ff65ab71c5cf1d8c38f51db6f05e3d60a45d3a3293c3248c925c25375921",
}

ETCD_VERSION = "3.4.13"
_ETCD_TARBALL_ARCH_SHA256 = {
    "amd64": "2ac029e47bab752dacdb7b30032f230f49e2f457cbc32e8f555c2210bb5ff107",
    "arm64": "1934ebb9f9f6501f706111b78e5e321a7ff8d7792d3d96a76e2d01874e42a300",
    "ppc64le": "fc77c3949b5178373734c3b276eb2281c954c3cd2225ccb05cdbdf721e1f775a",
}

# Dependencies needed for a Kubernetes "release", e.g. building docker images,
# debs, RPMs, or tarballs.
def release_dependencies():
    cni_tarballs()
    cri_tarballs()
    image_dependencies()
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
            urls = mirror("https://github.com/kubernetes-sigs/cri-tools/releases/download/v%s/crictl-v%s-%s.tar.gz" % (CRI_TOOLS_VERSION, CRI_TOOLS_VERSION, arch)),
        )

# Use skopeo to find these values: https://github.com/containers/skopeo
#
# Example
# Manifest: skopeo inspect docker://k8s.gcr.io/build-image/debian-base:buster-v1.8.0
# Arches: skopeo inspect --raw docker://k8s.gcr.io/build-image/debian-base:buster-v1.8.0
_DEBIAN_BASE_DIGEST = {
    "manifest": "sha256:22666783ee41fa619ad4d7ea40800bb40901d2e27d60c0ca3339a5851374763e",
    "amd64": "sha256:45965a68454706b7318a36cb9252c0e3f37a61b9e34578a56e06ac7d7ddb4d5e",
    "arm": "sha256:7e1ea4457b1a5969067d79b748d6e648834ef6523f153e0780213f21590ad3e8",
    "arm64": "sha256:336a612ad49a58e2440aa111fa3fc10e04c607b805debe4544fd2db6384d6ab8",
    "ppc64le": "sha256:046123ab9444d9c66132b179bed6954a8bd4e35c9ff0c2194c45979021f49655",
    "s390x": "sha256:468fae3b4ca48f0ea9c608994c12e99b32c9a617500c89efe98f84832a0ab007",
}

# Use skopeo to find these values: https://github.com/containers/skopeo
#
# Example
# Manifest: skopeo inspect docker://gcr.io/k8s-staging-build-image/debian-iptables:buster-v1.6.5
# Arches: skopeo inspect --raw docker://gcr.io/k8s-staging-build-image/debian-iptables:buster-v1.6.5
_DEBIAN_IPTABLES_DIGEST = {
    "manifest": "sha256:d226f3fd5f293ff513f53573a40c069b89d57d42338a1045b493bf702ac6b1f6",
    "amd64": "sha256:200be0a96b436ac42d50f04f291d51384001c0fb68f65836db6d18a0f6eca866",
    "arm": "sha256:4c705fd85f52162853df8cd5c38b445ae4090e02d9370257b45e104fb7dff070",
    "arm64": "sha256:7db471e96a33d4d1fc1611082a45f434e81b82402a1a3cd4255c6b5b2b9a5186",
    "ppc64le": "sha256:e83a0368cfe4e3b99f85b557e39bad55446ac9c14249337889998b59399905c9",
    "s390x": "sha256:1cff7805d2eda46bab962acd48a3cd8d536507149333d7c4706e57aad61b58b8",
}

# Use skopeo to find these values: https://github.com/containers/skopeo
#
# Example
# Manifest: skopeo inspect docker://k8s.gcr.io/build-image/go-runner:v2.3.1-go1.15.15-buster.0
# Arches: skopeo inspect --raw docker://k8s.gcr.io/build-image/go-runner:v2.3.1-go1.15.15-buster.0
_GO_RUNNER_DIGEST = {
    "manifest": "sha256:3677ad5ab7f58cb91b5bc44171d4b4ef20800c4730cac75e532d3b0b614001b7",
    "amd64": "sha256:f816d2e77334e456180c738628ce42dd15bec2725cccde8b3eb02b44664fa9f0",
    "arm": "sha256:f6ef14f21aa7e872afd2671fe72b3baf9440b9273e0053fcc41c408f57a82535",
    "arm64": "sha256:cf613b2b8fd1f3f940801b7e049699fbb0de5ba7c204ce57a525a869c91d9b1b",
    "ppc64le": "sha256:565eac823b7787acc2e6d67abc0c78ebac8e62b8aa20623def72db8317bc9d33",
    "s390x": "sha256:3520419608a21129627e1860621ae8d3b114b589153200fd01d20ef968cefa07",
}

def _digest(d, arch):
    if arch not in d:
        print("WARNING: %s not found in %r" % (arch, d))
        return d["manifest"]
    return d[arch]

def image_dependencies():
    for arch in SERVER_PLATFORMS["linux"]:
        container_pull(
            name = "go-runner-linux-" + arch,
            architecture = arch,
            digest = _digest(_GO_RUNNER_DIGEST, arch),
            registry = "k8s.gcr.io/build-image",
            repository = "go-runner",
            tag = "v2.3.1-go1.15.15-buster.0",  # ignored, but kept here for documentation
        )

        container_pull(
            name = "debian-base-" + arch,
            architecture = arch,
            digest = _digest(_DEBIAN_BASE_DIGEST, arch),
            registry = "k8s.gcr.io/build-image",
            repository = "debian-base",
            # Ensure the digests above are updated to match a new tag
            tag = "buster-v1.8.0",  # ignored, but kept here for documentation
        )

        container_pull(
            name = "debian-iptables-" + arch,
            architecture = arch,
            digest = _digest(_DEBIAN_IPTABLES_DIGEST, arch),
            registry = "k8s.gcr.io/build-image",
            repository = "debian-iptables",
            # Ensure the digests above are updated to match a new tag
            tag = "buster-v1.6.5",  # ignored, but kept here for documentation
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
