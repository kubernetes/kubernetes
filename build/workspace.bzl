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
# Manifest: skopeo inspect docker://gcr.io/k8s-staging-build-image/debian-base:v2.1.2
# Arches: skopeo inspect --raw docker://gcr.io/k8s-staging-build-image/debian-base:v2.1.2
_DEBIAN_BASE_DIGEST = {
    "manifest": "sha256:06190f821324ca89131437c51bdfe4b8515dc8ed26eba69c8f3698da227f5256",
    "amd64": "sha256:c9d003ca6daeaff7b1faf4356d010c67be310ff7b6c0af7b44f570571f15fc43",
    "arm": "sha256:52bda74a65e94166951ef5e33e6da847573e2a89dd6bc0bc3250a661adce1712",
    "arm64": "sha256:4e172c7b2a8ae261431e05509839c8b8c23a49ededed13af898c90896d466726",
    "ppc64le": "sha256:ad2190b49b3aebb1027b8fe685261c41ffbde170f49accc0816c296e59e9e927",
    "s390x": "sha256:813741dd7fc9585940f558ac142fd5121cf74d9c804ffa04e22a9df4da0f785d",
}

# Use skopeo to find these values: https://github.com/containers/skopeo
#
# Example
# Manifest: skopeo inspect docker://gcr.io/k8s-staging-build-image/debian-iptables:v12.1.1
# Arches: skopeo inspect --raw docker://gcr.io/k8s-staging-build-image/debian-iptables:v12.1.1
_DEBIAN_IPTABLES_DIGEST = {
    "manifest": "sha256:b92101e2910d9fe1415f35b6f7b696cf4a586f40f176ede019035bacc957a7a4",
    "amd64": "sha256:4b402df0d030b267f3dcdc6506d212d6b4c7fff6552b67f0c99be7ec1842f32a",
    "arm": "sha256:e976a5c88105db9a3ed462c2899aabbc5758f4f76e24098981010a226d344439",
    "arm64": "sha256:6da5b6aacb8c17615e1001aac0121517f541421036b20808a375f0dd79bf8564",
    "ppc64le": "sha256:63cf48fd9de1407302019ab524f7ed9ff7a05d9ecacc1a6ac77ee57fed21a078",
    "s390x": "sha256:be4039f250f26e1321d048e4dd6e81b78a80825c2dd490bacc0ac80ebb94bbc5",
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
            tag = "v2.1.2",  # ignored, but kept here for documentation
        )

        container_pull(
            name = "debian-iptables-" + arch,
            architecture = arch,
            digest = _digest(_DEBIAN_IPTABLES_DIGEST, arch),
            registry = "k8s.gcr.io/build-image",
            repository = "debian-iptables",
            # Ensure the digests above are updated to match a new tag
            tag = "v12.1.1",  # ignored, but kept here for documentation
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
