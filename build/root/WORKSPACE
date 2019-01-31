load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive", "http_file")
load("//build:workspace_mirror.bzl", "mirror")
load("//build:workspace.bzl", "CNI_VERSION", "CRI_TOOLS_VERSION")

http_archive(
    name = "bazel_skylib",
    sha256 = "eb5c57e4c12e68c0c20bc774bfbc60a568e800d025557bc4ea022c6479acc867",
    strip_prefix = "bazel-skylib-0.6.0",
    urls = mirror("https://github.com/bazelbuild/bazel-skylib/archive/0.6.0.tar.gz"),
)

load("@bazel_skylib//lib:versions.bzl", "versions")

versions.check(minimum_bazel_version = "0.18.0")

http_archive(
    name = "io_k8s_repo_infra",
    sha256 = "4ce2d8576e786c8fb8bc4f7ed79f5fda543568d78c09306d16cc8d1b7d5621f0",
    strip_prefix = "repo-infra-8a5707674e76b825bfd9a8624bae045bc6e5f24d",
    urls = mirror("https://github.com/kubernetes/repo-infra/archive/8a5707674e76b825bfd9a8624bae045bc6e5f24d.tar.gz"),
)

ETCD_VERSION = "3.3.10"

http_archive(
    name = "com_coreos_etcd",
    build_file = "@//third_party:etcd.BUILD",
    sha256 = "1620a59150ec0a0124a65540e23891243feb2d9a628092fb1edcc23974724a45",
    strip_prefix = "etcd-v%s-linux-amd64" % ETCD_VERSION,
    urls = mirror("https://github.com/coreos/etcd/releases/download/v%s/etcd-v%s-linux-amd64.tar.gz" % (ETCD_VERSION, ETCD_VERSION)),
)

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "492c3ac68ed9dcf527a07e6a1b2dcbf199c6bf8b35517951467ac32e421c06c1",
    urls = mirror("https://github.com/bazelbuild/rules_go/releases/download/0.17.0/rules_go-0.17.0.tar.gz"),
)

load("@io_bazel_rules_go//go:deps.bzl", "go_register_toolchains", "go_rules_dependencies")

go_rules_dependencies()

go_register_toolchains(
    go_version = "1.11.5",
)

http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "aed1c249d4ec8f703edddf35cbe9dfaca0b5f5ea6e4cd9e83e99f3b0d1136c3d",
    strip_prefix = "rules_docker-0.7.0",
    urls = mirror("https://github.com/bazelbuild/rules_docker/archive/v0.7.0.tar.gz"),
)

load(
    "@io_bazel_rules_docker//repositories:repositories.bzl",
    container_repositories = "repositories",
)

container_repositories()

load("@io_bazel_rules_docker//container:container.bzl", "container_pull")

http_file(
    name = "kubernetes_cni",
    downloaded_file_path = "kubernetes_cni.tgz",
    sha256 = "f04339a21b8edf76d415e7f17b620e63b8f37a76b2f706671587ab6464411f2d",
    urls = mirror("https://storage.googleapis.com/kubernetes-release/network-plugins/cni-plugins-amd64-v%s.tgz" % CNI_VERSION),
)

http_file(
    name = "cri_tools",
    downloaded_file_path = "cri_tools.tgz",
    sha256 = "e7d913bcce40bf54e37ab1d4b75013c823d0551e6bc088b217bc1893207b4844",
    urls = mirror("https://github.com/kubernetes-incubator/cri-tools/releases/download/v%s/crictl-v%s-linux-amd64.tar.gz" % (CRI_TOOLS_VERSION, CRI_TOOLS_VERSION)),
)

container_pull(
    name = "debian-base-amd64",
    digest = "sha256:8ccb65cd2dd7e0c24193d0742a20e4a673dbd11af5a33f16fcd471a31486866c",
    registry = "k8s.gcr.io",
    repository = "debian-base-amd64",
    tag = "0.4.1",  # ignored, but kept here for documentation
)

container_pull(
    name = "debian-iptables-amd64",
    digest = "sha256:9c41b4c326304b94eb96fdd2e181aa6e9995cc4642fcdfb570cedd73a419ba39",
    registry = "k8s.gcr.io",
    repository = "debian-iptables-amd64",
    tag = "v11.0.1",  # ignored, but kept here for documentation
)

container_pull(
    name = "debian-hyperkube-base-amd64",
    digest = "sha256:5d4ea2fb5fbe9a9a9da74f67cf2faefc881968bc39f2ac5d62d9167e575812a1",
    registry = "k8s.gcr.io",
    repository = "debian-hyperkube-base-amd64",
    tag = "0.12.1",  # ignored, but kept here for documentation
)

container_pull(
    name = "official_busybox",
    digest = "sha256:cb63aa0641a885f54de20f61d152187419e8f6b159ed11a251a09d115fdff9bd",
    registry = "index.docker.io",
    repository = "library/busybox",
    tag = "latest",  # ignored, but kept here for documentation
)

load("//build:workspace_mirror.bzl", "export_urls")

export_urls("workspace_urls")
