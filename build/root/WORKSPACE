http_archive(
    name = "io_bazel_rules_go",
    sha256 = "27a13726ff8621cfa4592fbef116b253043c2b093c8814265dc4489a0e67d229",
    strip_prefix = "rules_go-82483596ec203eb9c1849937636f4cbed83733eb",
    urls = ["https://github.com/bazelbuild/rules_go/archive/82483596ec203eb9c1849937636f4cbed83733eb.tar.gz"],
)

http_archive(
    name = "io_kubernetes_build",
    sha256 = "ca8fa1ee0928220d77fcaa6bcf40a26c57800c024e21b08c8dd9cc8fbf910236",
    strip_prefix = "repo-infra-0aafaab9e158d3628804242c6a9c4dd3eb8bce1f",
    urls = ["https://github.com/kubernetes/repo-infra/archive/0aafaab9e158d3628804242c6a9c4dd3eb8bce1f.tar.gz"],
)

ETCD_VERSION = "3.0.17"

new_http_archive(
    name = "com_coreos_etcd",
    build_file = "third_party/etcd.BUILD",
    sha256 = "274c46a7f8d26f7ae99d6880610f54933cbcf7f3beafa19236c52eb5df8c7a0b",
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
    sha256 = "40d780165c0b9fbb3ddca858df7347381af0e87e430c74863e4ce9d6f6441023",
    strip_prefix = "rules_docker-8359263f35227a3634ea023ff4ae163189eb4b26",
    urls = ["https://github.com/bazelbuild/rules_docker/archive/8359263f35227a3634ea023ff4ae163189eb4b26.tar.gz"],
)

load("@io_bazel_rules_go//go:def.bzl", "go_repositories")
load("@io_bazel_rules_docker//docker:docker.bzl", "docker_repositories", "docker_pull")

go_repositories(
    go_version = "1.8.3",
)

docker_repositories()

http_file(
    name = "kubernetes_cni",
    sha256 = "05ab3937bc68562e989dc143362ec4d4275262ba9f359338aed720fc914457a5",
    url = "https://storage.googleapis.com/kubernetes-release/network-plugins/cni-amd64-0799f5732f2a11b329d9e3d51b9c8f2e3759f2ff.tar.gz",
)

docker_pull(
    name = "debian-iptables-amd64",
    digest = "sha256:2e747bc7455b46350d8e57f05c03e109fa306861e7b2a2e8e1cd563932170cf1",
    registry = "gcr.io",
    repository = "google-containers/debian-iptables-amd64",
    tag = "v8",  # ignored, but kept here for documentation
)

docker_pull(
    name = "official_busybox",
    digest = "sha256:be3c11fdba7cfe299214e46edc642e09514dbb9bbefcd0d3836c05a1e0cd0642",
    registry = "index.docker.io",
    repository = "library/busybox",
    tag = "latest",  # ignored, but kept here for documentation
)
