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

CRI_TOOLS_VERSION = "1.20.0"
_CRI_TARBALL_ARCH_SHA256 = {
    "linux-386": "13ab9493cefca1d1ac5848ed52572e2ee5518a5bf2c527c0e5ed75b0e5c42c39",
    "linux-amd64": "44d5f550ef3f41f9b53155906e0229ffdbee4b19452b4df540265e29572b899c",
    "linux-arm": "ed5ffdd386261ec1146731421d4ac9c5c7f91e08486fee409452a3364bef792a",
    "linux-arm64": "eda6879710eb046d335162d4afe8494c6f8161142ad3188022852f64b92806a8",
    "linux-ppc64le": "da0c052983ba884f9605b14bf627664df67fcdb41c7f6908368bf4745f889b26",
    "linux-s390x": "88e1e41502e6f649e1a9dd0392d6ddec1854d6cd9d826b69d092e80d74fc4aca",
    "windows-386": "b37edede7e4eb11247f5677f4cab1a8bca4ea1bc26a5c6b3ee599adddc01f926",
    "windows-amd64": "cc909108ee84d39b2e9d7ac0cb9599b6fa7fc51f5a7da7014052684cd3e3f65e",
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
# Manifest: skopeo inspect docker://gcr.io/k8s-staging-build-image/debian-base:buster-v1.3.0
# Arches: skopeo inspect --raw docker://gcr.io/k8s-staging-build-image/debian-base:buster-v1.3.0
_DEBIAN_BASE_DIGEST = {
    "manifest": "sha256:d66137c7c362d1026dca670d1ff4c25e5b0770e8ace87ac3d008d52e4b0db338",
    "amd64": "sha256:a5ab028d9a730b78af9abb15b5db9b2e6f82448ab269d6f3a07d1834c571ccc6",
    "arm": "sha256:94e611363760607366ca1fed9375105b6c5fc922ab1249869b708690ca13733c",
    "arm64": "sha256:83512c52d44587271cd0f355c0a9a7e6c2412ddc66b8a8eb98f994277297a72f",
    "ppc64le": "sha256:9c8284b2797b114ebe8f3f1b2b5817a9c7f07f3f82513c49a30e6191a1acc1fc",
    "s390x": "sha256:d617637dd4df0bc1cfa524fae3b4892cfe57f7fec9402ad8dfa28e38e82ec688",
}

# Use skopeo to find these values: https://github.com/containers/skopeo
#
# Example
# Manifest: skopeo inspect docker://gcr.io/k8s-staging-build-image/debian-iptables:buster-v1.4.0
# Arches: skopeo inspect --raw docker://gcr.io/k8s-staging-build-image/debian-iptables:buster-v1.4.0
_DEBIAN_IPTABLES_DIGEST = {
    "manifest": "sha256:87f97cf2b62eb107871ee810f204ccde41affb70b29883aa898e93df85dea0f0",
    "amd64": "sha256:da837f39cf3af78adb796c0caa9733449ae99e51cf624590c328e4c9951ace7a",
    "arm": "sha256:bb6677337a4dbc3e578a3e87642d99be740dea391dc5e8987f04211c5e23abcd",
    "arm64": "sha256:6ad4717d69db2cc47bc2efc91cebb96ba736be1de49e62e0deffdbaf0fa2318c",
    "ppc64le": "sha256:168ccfeb861239536826a26da24ab5f68bb5349d7439424b7008b01e8f6534fc",
    "s390x": "sha256:5a88d4f4c29bac5b5c93195059b928f7346be11d0f0f7f6da0e14c0bfdbd1362",
}

# Use skopeo to find these values: https://github.com/containers/skopeo
#
# Example
# Manifest: skopeo inspect docker://gcr.io/k8s-staging-build-image/go-runner:buster-v2.2.4
# Arches: skopeo inspect --raw docker://gcr.io/k8s-staging-build-image/go-runner:buster-v2.2.4
_GO_RUNNER_DIGEST = {
    "manifest": "sha256:059fd64240ad0fcd0f6eee6a791004747f1b4a5d3c44dd5ca68258bda4555e67",
    "amd64": "sha256:66cdf0be9ba12e3183c43664f9f02b14e2642b19a3d82784d15f56d01e521017",
    "arm": "sha256:61eefb4de21f946c1d4bc8e941f17a2a77c38f66c8b4a07d6da6d1e8918d47c4",
    "arm64": "sha256:4b8dcdf86964ae8ceea7eca409ca7b87ad01421c1db0a0a99cc99676785496e3",
    "ppc64le": "sha256:1e9368e5a81184eb1fd17053b7796f20663724c2046d17ce394b40f5e47affb1",
    "s390x": "sha256:9ed1b51936ff6057d1534f5d22309d4e7369211d302b3bd72554e9d3963dbae3",
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
            tag = "buster-v2.2.4",  # ignored, but kept here for documentation
        )

        container_pull(
            name = "debian-base-" + arch,
            architecture = arch,
            digest = _digest(_DEBIAN_BASE_DIGEST, arch),
            registry = "k8s.gcr.io/build-image",
            repository = "debian-base",
            # Ensure the digests above are updated to match a new tag
            tag = "buster-v1.3.0",  # ignored, but kept here for documentation
        )

        container_pull(
            name = "debian-iptables-" + arch,
            architecture = arch,
            digest = _digest(_DEBIAN_IPTABLES_DIGEST, arch),
            registry = "k8s.gcr.io/build-image",
            repository = "debian-iptables",
            # Ensure the digests above are updated to match a new tag
            tag = "buster-v1.4.0",  # ignored, but kept here for documentation
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
