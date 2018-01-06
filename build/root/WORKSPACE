http_archive(
    name = "io_bazel_rules_go",
    sha256 = "e8c7f1fda9ee482745a5b35e8314ac3ae744d4ba30f3e6de28148fd166044306",
    strip_prefix = "rules_go-737df20c53499fd84b67f04c6ca9ccdee2e77089",
    urls = ["https://github.com/bazelbuild/rules_go/archive/737df20c53499fd84b67f04c6ca9ccdee2e77089.tar.gz"],
)

http_archive(
    name = "io_kubernetes_build",
    sha256 = "cf138e48871629345548b4aaf23101314b5621c1bdbe45c4e75edb45b08891f0",
    strip_prefix = "repo-infra-1fb0a3ff0cc5308a6d8e2f3f9c57d1f2f940354e",
    urls = ["https://github.com/kubernetes/repo-infra/archive/1fb0a3ff0cc5308a6d8e2f3f9c57d1f2f940354e.tar.gz"],
)

ETCD_VERSION = "3.1.10"

new_http_archive(
    name = "com_coreos_etcd",
    build_file = "third_party/etcd.BUILD",
    sha256 = "2d335f298619c6fb02b1124773a56966e448ad9952b26fea52909da4fe80d2be",
    strip_prefix = "etcd-v%s-linux-amd64" % ETCD_VERSION,
    urls = ["https://github.com/coreos/etcd/releases/download/v%s/etcd-v%s-linux-amd64.tar.gz" % (ETCD_VERSION, ETCD_VERSION)],
)

# This contains a patch to not prepend ./ to tarfiles produced by pkg_tar.
# When merged upstream, we'll no longer need to use ixdy's fork:
# https://bazel-review.googlesource.com/#/c/10390/
http_archive(
    name = "io_bazel",
    sha256 = "892a84aa1e7c1f99fb57bb056cb648745c513077252815324579a012d263defb",
    strip_prefix = "bazel-df2c687c22bdd7c76f3cdcc85f38fefd02f0b844",
    urls = ["https://github.com/ixdy/bazel/archive/df2c687c22bdd7c76f3cdcc85f38fefd02f0b844.tar.gz"],
)

http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "c440717ee9b1b2f4a1e9bf5622539feb5aef9db83fc1fa1517818f13c041b0be",
    strip_prefix = "rules_docker-8bbe2a8abd382641e65ff7127a3700a8530f02ce",
    urls = ["https://github.com/bazelbuild/rules_docker/archive/8bbe2a8abd382641e65ff7127a3700a8530f02ce.tar.gz"],
)

load("@io_kubernetes_build//defs:bazel_version.bzl", "check_version")

check_version("0.8.0")

load("@io_bazel_rules_go//go:def.bzl", "go_rules_dependencies", "go_register_toolchains", "go_download_sdk")
load("@io_bazel_rules_docker//docker:docker.bzl", "docker_repositories", "docker_pull")

go_rules_dependencies()

go_register_toolchains(
    go_version = "1.9.2",
)

docker_repositories()

http_file(
    name = "kubernetes_cni",
    sha256 = "f04339a21b8edf76d415e7f17b620e63b8f37a76b2f706671587ab6464411f2d",
    url = "https://storage.googleapis.com/kubernetes-release/network-plugins/cni-plugins-amd64-v0.6.0.tgz",
)

docker_pull(
    name = "debian-iptables-amd64",
    digest = "sha256:a3b936c0fb98a934eecd2cfb91f73658d402b29116084e778ce9ddb68e55383e",
    registry = "gcr.io",
    repository = "google-containers/debian-iptables-amd64",
    tag = "v10",  # ignored, but kept here for documentation
)

docker_pull(
    name = "debian-hyperkube-base-amd64",
    digest = "sha256:fc1b461367730660ac5a40c1eb2d1b23221829acf8a892981c12361383b3742b",
    registry = "gcr.io",
    repository = "google-containers/debian-hyperkube-base-amd64",
    tag = "0.8",  # ignored, but kept here for documentation
)

docker_pull(
    name = "official_busybox",
    digest = "sha256:be3c11fdba7cfe299214e46edc642e09514dbb9bbefcd0d3836c05a1e0cd0642",
    registry = "index.docker.io",
    repository = "library/busybox",
    tag = "latest",  # ignored, but kept here for documentation
)
