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
# Manifest: skopeo inspect docker://gcr.io/k8s-staging-build-image/debian-hyperkube-base:v1.1.3
# Arches: skopeo inspect --raw docker://gcr.io/k8s-staging-build-image/debian-hyperkube-base:v1.1.3
_DEBIAN_HYPERKUBE_BASE_DIGEST = {
    "manifest": "sha256:9190cda1ba1c1bf07eace001240d648a1d480bc55cc383b49f4d29d4f9ce6a51",
    "amd64": "sha256:9f1f43babd36bfd2cce05a63b0331ed580c1fea876c833aae7036ffa5c6346c0",
    "arm": "sha256:d4d0f188ee723c3f276922512ce2d335af4a54caf65b5df5e9105955d34a1d72",
    "arm64": "sha256:9b98a76c8bca6d4b19adae1a20d475d959428a28530febba8c044ca0aabb471d",
    "ppc64le": "sha256:0ba61ab0ec8b2ed9501e4e1245119fd8af1c211eab0cf3bfddb1dbd95ed7ef08",
    "s390x": "sha256:f031fe21b338443166ae00142d05601507f9959d1bce94b5d4f0dbe5cd75f64b",
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
            tag = "v2.1.3",  # ignored, but kept here for documentation
        )

        container_pull(
            name = "debian-iptables-" + arch,
            architecture = arch,
            digest = _digest(_DEBIAN_IPTABLES_DIGEST, arch),
            registry = "us.gcr.io/k8s-artifacts-prod/build-image",
            repository = "debian-iptables",
            # Ensure the digests above are updated to match a new tag
            tag = "v12.1.2",  # ignored, but kept here for documentation
        )

        container_pull(
            name = "debian-hyperkube-base-" + arch,
            architecture = arch,
            digest = _digest(_DEBIAN_HYPERKUBE_BASE_DIGEST, arch),
            registry = "us.gcr.io/k8s-artifacts-prod/build-image",
            repository = "debian-hyperkube-base",
            # Ensure the digests above are updated to match a new tag
            tag = "v1.1.3",  # ignored, but kept here for documentation
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
