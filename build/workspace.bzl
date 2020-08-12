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

CRI_TOOLS_VERSION = "1.18.0"
_CRI_TARBALL_ARCH_SHA256 = {
    "linux-386": "a1aaf482928d0a19aabeb321e406333c5ddecf77a532f7ec8c0bd6ca7014101e",
    "linux-amd64": "876dd2b3d0d1c2590371f940fb1bf1fbd5f15aebfbe456703ee465d959700f4a",
    "linux-arm": "d420925d10b47a234b7e51e9cf1039c3c09f2703945a99435549fcdd7487ae3a",
    "linux-arm64": "95ba32c47ad690b1e3e24f60255273dd7d176e62b1a0b482e5b44a7c31639979",
    "linux-ppc64le": "53a1fedbcee37f5d6c9480d21a9bb17f1c0214ffe7b640e39231a59927a665ef",
    "linux-s390x": "114c8885a7eeb43bbe19baaf23c04a5761d06330ba8e7aa39a3a15c2051221f1",
    "windows-386": "f37e8b5c499fb5a2bd06668782a7dc34e5acf2fda6d1bfe8f0ea9c773359a378",
    "windows-amd64": "5045bcc6d8b0e6004be123ab99ea06e5b1b2ae1e586c968fcdf85fccd4d67ae1",
}

ETCD_VERSION = "3.4.9"
_ETCD_TARBALL_ARCH_SHA256 = {
    "amd64": "bcab421f6bf4111accfceb004e0a0ac2bcfb92ac93081d9429e313248dd78c41",
    "arm64": "fd9bf37662a851905d75160fea0f5d10055c1bee0a734e78c5112cc56c9faa18",
    "ppc64le": "bfdcea0fc83c6d6edb70667a2272f8fc597c61976ecc6f8ecbfeb380ff02618b",
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
            urls = mirror("https://github.com/kubernetes-incubator/cri-tools/releases/download/v%s/crictl-v%s-%s.tar.gz" % (CRI_TOOLS_VERSION, CRI_TOOLS_VERSION, arch)),
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
# Manifest: skopeo inspect docker://gcr.io/k8s-staging-build-image/go-runner:v0.1.1
# Arches: skopeo inspect --raw docker://gcr.io/k8s-staging-build-image/go-runner:v0.1.1
_GO_RUNNER_DIGEST = {
    "manifest": "sha256:4892faa2de0533bc1af72b9b233936f21a9e7362063345d170de1a8f464f2ad8",
    "amd64": "sha256:821e48a96d46aa53d2f7f5ef9d9093ed69979957a0a7092d1c09c44d81028a9d",
    "arm": "sha256:2cc042179887b6baa0792e156b53f4cb94181b1a99153790402bd8e517e8cf56",
    "arm64": "sha256:00ca7f34275349330a5d8ddffd15e2980fe5b2cbdd410f063f4e7617e0e71c29",
    "ppc64le": "sha256:3e25e0d0e9d17033f3e86d4af5787c7fc5f1173e174d77eebdc14df1a06f1c99",
    "s390x": "sha256:3e34e290cd35a90285991a575e2e79fddfb161c66f13bc5662a1cc0a4ade32e0",
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
            tag = "v0.1.1",  # ignored, but kept here for documentation
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
