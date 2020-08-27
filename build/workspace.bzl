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
# Manifest: skopeo inspect docker://gcr.io/k8s-staging-build-image/debian-base:v2.1.3
# Arches: skopeo inspect --raw docker://gcr.io/k8s-staging-build-image/debian-base:v2.1.3
_DEBIAN_BASE_DIGEST = {
    "manifest": "sha256:37cfe133a6ff3fc3aa4aa5ab9fda127861902940b8e8078fff7191c7a81be8d8",
    "amd64": "sha256:dc06e242160076b72bd75135fb3dd0a9e91f386b2d812ec10cbf9e65864c755d",
    "arm": "sha256:9c52e9b31d679102586381fb4a03bba73fc05992adacce31df3db6d2f75f010e",
    "arm64": "sha256:9a3ae250d59f317c9cf1f7bf0b61c0d90e70012ee82996867655401968076ee4",
    "ppc64le": "sha256:d7c50e06d954bedb1f362ce42380f7a059423f8cbd9e7b426a7f2d73dfb4431a",
    "s390x": "sha256:7e6d43baf4972f7faa33f1179fb92ff858a3e0e23f78b96a5821d13199b9da91",
}

# Use skopeo to find these values: https://github.com/containers/skopeo
#
# Example
# Manifest: skopeo inspect docker://gcr.io/k8s-staging-build-image/debian-iptables:v12.1.2
# Arches: skopeo inspect --raw docker://gcr.io/k8s-staging-build-image/debian-iptables:v12.1.2
_DEBIAN_IPTABLES_DIGEST = {
    "manifest": "sha256:fff1fd5ab38fefde8c9eee5470ed6ea737f30e5ab86e1c0c0d429fa6add28a84",
    "amd64": "sha256:448c0e019689da3f4e238922824551d02578473f7b5d11604fffd528056caafa",
    "arm": "sha256:08e267a6297640eb0b218a783cabaef0a039cc4734b4a2d9cb54ee41cd82656a",
    "arm64": "sha256:a83cf1d501ad33f5aa93e2719baa59b054939b8a819c3997f915a6acfaa8e31a",
    "ppc64le": "sha256:c86649ac541c35431a4df8ba44e7c61923d9e34679fd17da7723db7b6ecc5245",
    "s390x": "sha256:53437fe32e13bb0bd2af08ddaaf2e908add8ab576574d881ba79544f34a774b2",
}

# Use skopeo to find these values: https://github.com/containers/skopeo
#
# Example
# Manifest: skopeo inspect docker://gcr.io/k8s-staging-build-image/go-runner:buster-v2.0.0
# Arches: skopeo inspect --raw docker://gcr.io/k8s-staging-build-image/go-runner:buster-v2.0.0
_GO_RUNNER_DIGEST = {
    "manifest": "sha256:ff6e2f3683e7d284674ed18341fc898060204e8c43c9b477e13c6f7faf3e66d4",
    "amd64": "sha256:140404aed601b95a2a0a1aeac0608a0fdbd5fc339a8ea6b2ee4a63c7e1f56415",
    "arm": "sha256:5d4e8c77bc472610e7e46bbd2b83e167e243434b8287ba2ffe6b09aba9f08ecc",
    "arm64": "sha256:62429a05973522064480deb44134e3ca355ee89c7781f3fc3ee9072f17de0085",
    "ppc64le": "sha256:05c8575486ccea90c35e8d8ba28c84aee57a03d58329b1354cf7193c372d2de2",
    "s390x": "sha256:e886ab4557e60293081f2e47a5b52e84bd3d60290a0f46fb99fac6eec35479ec",
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
            tag = "buster-v2.0.0",  # ignored, but kept here for documentation
        )

        container_pull(
            name = "debian-base-" + arch,
            architecture = arch,
            digest = _digest(_DEBIAN_BASE_DIGEST, arch),
            registry = "k8s.gcr.io/build-image",
            repository = "debian-base",
            # Ensure the digests above are updated to match a new tag
            tag = "v2.1.3",  # ignored, but kept here for documentation
        )

        container_pull(
            name = "debian-iptables-" + arch,
            architecture = arch,
            digest = _digest(_DEBIAN_IPTABLES_DIGEST, arch),
            registry = "k8s.gcr.io/build-image",
            repository = "debian-iptables",
            # Ensure the digests above are updated to match a new tag
            tag = "v12.1.2",  # ignored, but kept here for documentation
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
