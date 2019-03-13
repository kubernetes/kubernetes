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

CNI_VERSION = "0.6.0"
_CNI_TARBALL_ARCH_SHA256 = {
    "amd64": "f04339a21b8edf76d415e7f17b620e63b8f37a76b2f706671587ab6464411f2d",
    "arm": "ffb62021d2fc6e1266dc6ef7f2058125b6e6b44c016291a2b04a15ed9b4be70a",
    "arm64": "016bbc989877e35e3cd49fafe11415fb2717e52c74fde6b1650411154cb91b81",
    "ppc64le": "dd38dec69b167cfe40ecbba4b18cfe5b4296f2e49b90c00804b3988ef968e859",
    "s390x": "7708289eee7e52ad055407c421033d8e593f5cf1a0b43a872f09eb4e1508aafc",
}

CRI_TOOLS_VERSION = "1.12.0"
_CRI_TARBALL_ARCH_SHA256 = {
    "amd64": "e7d913bcce40bf54e37ab1d4b75013c823d0551e6bc088b217bc1893207b4844",
    "arm": "ca6b4ac80278d32d9cc8b8b19de140fd1cc35640f088969f7068fea2df625490",
    "arm64": "8466f08b59bf36d2eebcb9428c3d4e6e224c3065d800ead09ad730ce374da6fe",
    "ppc64le": "ec6254f1f6ffa064ba41825aab5612b7b005c8171fbcdac2ca3927d4e393000f",
    "s390x": "814aa9cd496be416612c2653097a1c9eb5784e38aa4889034b44ebf888709057",
}

ETCD_VERSION = "3.3.10"
_ETCD_TARBALL_ARCH_SHA256 = {
    "amd64": "1620a59150ec0a0124a65540e23891243feb2d9a628092fb1edcc23974724a45",
    "arm64": "5ec97b0b872adce275b8130d19db314f7f2b803aeb24c4aae17a19e2d66853c4",
    "ppc64le": "148fe96f0ec1813c5db9916199e96a913174304546bc8447a2d2f9fee4b8f6c2",
}

# Note that these are digests for the manifest list. We resolve the manifest
# list to each of its platform-specific images in
# debian_image_dependencies().
_DEBIAN_BASE_DIGEST = "sha256:6966a0aedd7592c18ff2dd803c08bd85780ee19f5e3a2e7cf908a4cd837afcde"  # 0.4.1
_DEBIAN_IPTABLES_DIGEST = "sha256:656e45c00083359107b1d6ae0411ff3894ba23011a8533e229937a71be84e063"  # v11.0.1
_DEBIAN_HYPERKUBE_BASE_DIGEST = "sha256:8cabe02be6e86685d8860b7ace7c7addc9591a339728703027a4854677f1c772"  # 0.12.1

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

def debian_image_dependencies():
    for arch in SERVER_PLATFORMS["linux"]:
        container_pull(
            name = "debian-base-" + arch,
            architecture = arch,
            digest = _DEBIAN_BASE_DIGEST,
            registry = "k8s.gcr.io",
            repository = "debian-base",
        )

        container_pull(
            name = "debian-iptables-" + arch,
            architecture = arch,
            digest = _DEBIAN_IPTABLES_DIGEST,
            registry = "k8s.gcr.io",
            repository = "debian-iptables",
        )

        container_pull(
            name = "debian-hyperkube-base-" + arch,
            architecture = arch,
            digest = _DEBIAN_HYPERKUBE_BASE_DIGEST,
            registry = "k8s.gcr.io",
            repository = "debian-hyperkube-base",
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
