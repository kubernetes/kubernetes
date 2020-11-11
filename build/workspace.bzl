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
# Manifest: skopeo inspect docker://gcr.io/k8s-staging-build-image/debian-base:buster-v1.2.0
# Arches: skopeo inspect --raw docker://gcr.io/k8s-staging-build-image/debian-base:buster-v1.2.0
_DEBIAN_BASE_DIGEST = {
    "manifest": "sha256:ea668d3febd312e0edfbbdab6bd7d86448ddc8fddb484e20ec76b36a7aeac04c",
    "amd64": "sha256:2f3e61995bcd4b3a1a0aef49e4a7a6817c978031734b09df2aaaa28181898b0e",
    "arm": "sha256:d1073dcf8f1d55fbbd297e5b280375b6f276ea83a08a25fd59dc4f3ca43c6d50",
    "arm64": "sha256:dadcff1ab81177de4914f6db0e7d78a52e525daf7a1efb246cb3545de5e818d1",
    "ppc64le": "sha256:bfb24dc0d1e71e1deb0f04a078fadf2c94070266746b1b5acc4e739aa57d5601",
    "s390x": "sha256:cfe6a3508b7ee198cb5a0b3a62e0981676b1dfa4b3049f36398d03e6bd35a801",
}

# Use skopeo to find these values: https://github.com/containers/skopeo
#
# Example
# Manifest: skopeo inspect docker://gcr.io/k8s-staging-build-image/debian-iptables:buster-v1.3.0
# Arches: skopeo inspect --raw docker://gcr.io/k8s-staging-build-image/debian-iptables:buster-v1.3.0
_DEBIAN_IPTABLES_DIGEST = {
    "manifest": "sha256:4c9410a4ee555dcb0e8b7bd6fc77c65ac400f7c5bd4555df8187630efaea6ea4",
    "amd64": "sha256:e30919918299988b318f0208e7fd264dee21a6be9d74bbd9f7fc15e78eade9b4",
    "arm": "sha256:bf59578f532bfd3378c4a713eeb14cf0dbed224d5ad03f549165f8d853997ca4",
    "arm64": "sha256:3d7ede6013b0516f1ec3852590895d4a7b6ec8f5e15bebc1a55237bba4538da2",
    "ppc64le": "sha256:ebd3bb280f8da8fc6a7158b7c2fc59b4552487bffd95d0b5ac1b190aff7b0fd9",
    "s390x": "sha256:a73b94aace7a571f36149aa917d4c7ee13453ed24a31f26549dd13b386dae4c1",
}

# Use skopeo to find these values: https://github.com/containers/skopeo
#
# Example
# Manifest: skopeo inspect docker://gcr.io/k8s-staging-build-image/go-runner:buster-v2.0.1
# Arches: skopeo inspect --raw docker://gcr.io/k8s-staging-build-image/go-runner:buster-v2.0.1
_GO_RUNNER_DIGEST = {
    "manifest": "sha256:687c17db2f5cd4aea13faa7ae56bee639a5b11f380c431a9800205624f53541c",
    "amd64": "sha256:b02bdb3444b1e7fb14cb5b60174f0e8f0a087ff4c294352e6c31c17da99a4ee2",
    "arm": "sha256:0d7563c814c0cd88bc5937b6e606d266409b5b7cee2deb6c04c6dcb6d7daaa5d",
    "arm64": "sha256:78f42645ddfd2ab9dfc4053834aa0042c82c8c550f9f61a2a76fd9f1791e5308",
    "ppc64le": "sha256:93e3ca63df801a5c1ad15bdbb9c50fa38e5db3479a92d8f4516c00dfd736f227",
    "s390x": "sha256:d7ed7bd8d58a6570504f14a50d502c2df97f944378f9f5306519f3379cb92fe2",
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
            tag = "buster-v2.0.1",  # ignored, but kept here for documentation
        )

        container_pull(
            name = "debian-base-" + arch,
            architecture = arch,
            digest = _digest(_DEBIAN_BASE_DIGEST, arch),
            registry = "k8s.gcr.io/build-image",
            repository = "debian-base",
            # Ensure the digests above are updated to match a new tag
            tag = "buster-v1.2.0",  # ignored, but kept here for documentation
        )

        container_pull(
            name = "debian-iptables-" + arch,
            architecture = arch,
            digest = _digest(_DEBIAN_IPTABLES_DIGEST, arch),
            registry = "k8s.gcr.io/build-image",
            repository = "debian-iptables",
            # Ensure the digests above are updated to match a new tag
            tag = "buster-v1.3.0",  # ignored, but kept here for documentation
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
