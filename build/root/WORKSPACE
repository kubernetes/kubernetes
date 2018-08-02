load("//build:workspace_mirror.bzl", "mirror")
load("//build:workspace.bzl", "CRI_TOOLS_VERSION")

http_archive(
    name = "io_bazel_rules_go",
    sha256 = "8b68d0630d63d95dacc0016c3bb4b76154fe34fca93efd65d1c366de3fcb4294",
    urls = mirror("https://github.com/bazelbuild/rules_go/releases/download/0.12.1/rules_go-0.12.1.tar.gz"),
)

http_archive(
    name = "io_kubernetes_build",
    sha256 = "6d87da8d97ccac3702eb9894541c32dd19a312f783f863e44bf8262d05dfaf2e",
    strip_prefix = "repo-infra-3c350a455362b622fe786e63f8f07b2a87f54f7b",
    urls = mirror("https://github.com/kubernetes/repo-infra/archive/3c350a455362b622fe786e63f8f07b2a87f54f7b.tar.gz"),
)

http_archive(
    name = "bazel_skylib",
    sha256 = "bbccf674aa441c266df9894182d80de104cabd19be98be002f6d478aaa31574d",
    strip_prefix = "bazel-skylib-2169ae1c374aab4a09aa90e65efe1a3aad4e279b",
    urls = mirror("https://github.com/bazelbuild/bazel-skylib/archive/2169ae1c374aab4a09aa90e65efe1a3aad4e279b.tar.gz"),
)

ETCD_VERSION = "3.2.18"

new_http_archive(
    name = "com_coreos_etcd",
    build_file = "third_party/etcd.BUILD",
    sha256 = "b729db0732448064271ea6fdcb901773c4fe917763ca07776f22d0e5e0bd4097",
    strip_prefix = "etcd-v%s-linux-amd64" % ETCD_VERSION,
    urls = mirror("https://github.com/coreos/etcd/releases/download/v%s/etcd-v%s-linux-amd64.tar.gz" % (ETCD_VERSION, ETCD_VERSION)),
)

http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "c440717ee9b1b2f4a1e9bf5622539feb5aef9db83fc1fa1517818f13c041b0be",
    strip_prefix = "rules_docker-8bbe2a8abd382641e65ff7127a3700a8530f02ce",
    urls = mirror("https://github.com/bazelbuild/rules_docker/archive/8bbe2a8abd382641e65ff7127a3700a8530f02ce.tar.gz"),
)

load("@bazel_skylib//:lib.bzl", "versions")

versions.check(minimum_bazel_version = "0.13.0")

load("@io_bazel_rules_go//go:def.bzl", "go_download_sdk", "go_register_toolchains", "go_rules_dependencies")
load("@io_bazel_rules_docker//docker:docker.bzl", "docker_pull", "docker_repositories")

go_rules_dependencies()

go_register_toolchains(
    go_version = "1.10.3",
)

docker_repositories()

http_file(
    name = "kubernetes_cni",
    sha256 = "f04339a21b8edf76d415e7f17b620e63b8f37a76b2f706671587ab6464411f2d",
    urls = mirror("https://storage.googleapis.com/kubernetes-release/network-plugins/cni-plugins-amd64-v0.6.0.tgz"),
)

http_file(
    name = "cri_tools",
    sha256 = "ccf83574556793ceb01717dc91c66b70f183c60c2bbec70283939aae8fdef768",
    urls = mirror("https://github.com/kubernetes-incubator/cri-tools/releases/download/v%s/crictl-v%s-linux-amd64.tar.gz" % (CRI_TOOLS_VERSION, CRI_TOOLS_VERSION)),
)

docker_pull(
    name = "debian-iptables-amd64",
    digest = "sha256:fb18678f8203ca1bd2fad2671e3ebd80cb408a1baae423d4ad39c05f4caac4e1",
    registry = "k8s.gcr.io",
    repository = "debian-iptables-amd64",
    tag = "v10",  # ignored, but kept here for documentation
)

docker_pull(
    name = "debian-hyperkube-base-amd64",
    digest = "sha256:cc782ed16599000ca4c85d47ec6264753747ae1e77520894dca84b104a7621e2",
    registry = "k8s.gcr.io",
    repository = "debian-hyperkube-base-amd64",
    tag = "0.10",  # ignored, but kept here for documentation
)

docker_pull(
    name = "official_busybox",
    digest = "sha256:4cee1979ba0bf7db9fc5d28fb7b798ca69ae95a47c5fecf46327720df4ff352d",
    registry = "index.docker.io",
    repository = "library/busybox",
    tag = "latest",  # ignored, but kept here for documentation
)

load("//build:workspace_mirror.bzl", "export_urls")

export_urls("workspace_urls")
