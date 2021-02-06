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
# Manifest: skopeo inspect docker://gcr.io/k8s-staging-build-image/debian-base:buster-v1.4.0
# Arches: skopeo inspect --raw docker://gcr.io/k8s-staging-build-image/debian-base:buster-v1.4.0
_DEBIAN_BASE_DIGEST = {
    "manifest": "sha256:36652ef8e4dd6715de02e9b68e5c122ed8ee06c75f83f5c574b97301e794c3fb",
    "amd64": "sha256:afff10fcd513483e492807f8d934bdf0be4a237997f55e0f1f8e34c04a6cb213",
    "arm": "sha256:27e6e66ea3c4c4ca6dbfc8c949f0c4c870f038f4500fd267c242422a244f233c",
    "arm64": "sha256:4333a5edc9ce6d6660c76104749c2e50e6158e57c8e5956f732991bb032a8ce1",
    "ppc64le": "sha256:01a0ba2645883ea8d985460c2913070a90a098056cc6d188122942678923ddb7",
    "s390x": "sha256:610526b047d4b528d9e14b4f15347aa4e37af0c47e1307a2f7aebf8745c8a323",
}

# Use skopeo to find these values: https://github.com/containers/skopeo
#
# Example
# Manifest: skopeo inspect docker://gcr.io/k8s-staging-build-image/debian-iptables:buster-v1.5.0
# Arches: skopeo inspect --raw docker://gcr.io/k8s-staging-build-image/debian-iptables:buster-v1.5.0
_DEBIAN_IPTABLES_DIGEST = {
    "manifest": "sha256:abe8cef9e116f2d5ec1175c386e33841ff3386779138b425af876384b0fd7ccb",
    "amd64": "sha256:b4b8b1e0d4617011dd03f20b804cc2e50bf48bafc36b1c8c7bd23fd44bfd641e",
    "arm": "sha256:09f79b3a00268705a8f8462f1528fed536e204905359f21e9965f08dd306c60a",
    "arm64": "sha256:b4fa11965f34a9f668c424b401c0af22e88f600d22c899699bdb0bd1e6953ad6",
    "ppc64le": "sha256:0ea0be4dec281b506f6ceef4cb3594cabea8d80e2dc0d93c7eb09d46259dd837",
    "s390x": "sha256:50ef25fba428b6002ef0a9dea7ceae5045430dc1035d50498a478eefccba17f5",
}

# Use skopeo to find these values: https://github.com/containers/skopeo
#
# Example
# Manifest: skopeo inspect docker://gcr.io/k8s-staging-build-image/go-runner:buster-v2.2.4
# Arches: skopeo inspect --raw docker://gcr.io/k8s-staging-build-image/go-runner:buster-v2.3.1
_GO_RUNNER_DIGEST = {
    "manifest": "sha256:cd45714e4824eeff6f107d9e3b4f79be9ee0cf5071dc46caf755d3f324a36089",
    "amd64": "sha256:309379049147b749d2bc63cd8bb2d6c46a68f45fd7fc5fd391d221b42e2c7196",
    "arm": "sha256:81ad4220d42a19e5e11ccb4b385b404ab287d6417f9b51077ea15df5196d6e75",
    "arm64": "sha256:93ccd74b2a434e21cd150cf89b10c6fc5e0bf66691ee5c8f22bf1241d168c445",
    "ppc64le": "sha256:4a7f8dce0f4505e43790fb660b67f4cebad91fae1835c79d0132ba6ecf480701",
    "s390x": "sha256:e6fa60bd53c8f3706c4d1cd6cd6bc3e95d01b4a924daab004fca9bf403b03e41",
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
            tag = "buster-v2.3.1",  # ignored, but kept here for documentation
        )

        container_pull(
            name = "debian-base-" + arch,
            architecture = arch,
            digest = _digest(_DEBIAN_BASE_DIGEST, arch),
            registry = "k8s.gcr.io/build-image",
            repository = "debian-base",
            # Ensure the digests above are updated to match a new tag
            tag = "buster-v1.4.0",  # ignored, but kept here for documentation
        )

        container_pull(
            name = "debian-iptables-" + arch,
            architecture = arch,
            digest = _digest(_DEBIAN_IPTABLES_DIGEST, arch),
            registry = "k8s.gcr.io/build-image",
            repository = "debian-iptables",
            # Ensure the digests above are updated to match a new tag
            tag = "buster-v1.5.0",  # ignored, but kept here for documentation
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
