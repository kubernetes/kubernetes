http_archive(
    name = "io_bazel_rules_go",
    sha256 = "64294fd0e74d2aafa03ec3a1f2f9c167e27d17c9a5cf393e8bf79e43258de73d",
    strip_prefix = "rules_go-a9df110cf04e167b33f10473c7e904d780d921e6",
    urls = ["https://github.com/bazelbuild/rules_go/archive/a9df110cf04e167b33f10473c7e904d780d921e6.tar.gz"],
)

http_archive(
    name = "io_kubernetes_build",
    sha256 = "8d1cff71523565996903076cec6cad8424afa6eb93a342d0d810a55c911e23c7",
    strip_prefix = "repo-infra-61b7247ebf472398bdea148d8f67e3a1849d6de9",
    urls = ["https://github.com/kubernetes/repo-infra/archive/61b7247ebf472398bdea148d8f67e3a1849d6de9.tar.gz"],
)

# This contains a patch to not prepend ./ to tarfiles produced by pkg_tar.
# When merged upstream, we'll no longer need to use ixdy's fork:
# https://bazel-review.googlesource.com/#/c/10390/
http_archive(
    name = "io_bazel",
    sha256 = "667d32da016b1e2f63cf345cd3583989ec4a165034df383a01996d93635753a0",
    strip_prefix = "bazel-df2c687c22bdd7c76f3cdcc85f38fefd02f0b844",
    urls = ["https://github.com/ixdy/bazel/archive/df2c687c22bdd7c76f3cdcc85f38fefd02f0b844.tar.gz"],
)

http_archive(
    name = "io_bazel_rules_docker",
    sha256 = "261fbd8fda1d06a12a0479019b46acd302c6aaa8df8e49383dc37917f20492a1",
    strip_prefix = "rules_docker-52d9faf209ff6d16eb850b6b66d03483735e0633",
    urls = ["https://github.com/bazelbuild/rules_docker/archive/52d9faf209ff6d16eb850b6b66d03483735e0633.tar.gz"],
)

load("@io_bazel_rules_go//go:def.bzl", "go_repositories")
load("@io_bazel_rules_docker//docker:docker.bzl", "docker_repositories", "docker_pull")

go_repositories(
    go_version = "1.8.3",
)

docker_repositories()

# for building docker base images
debs = (
    (
        "busybox_deb",
        "5f81f140777454e71b9e5bfdce9c89993de5ddf4a7295ea1cfda364f8f630947",
        "http://ftp.us.debian.org/debian/pool/main/b/busybox/busybox-static_1.22.0-19+b3_amd64.deb",
        "https://storage.googleapis.com/kubernetes-release/debs/busybox-static_1.22.0-19+b3_amd64.deb",
    ),
)

[http_file(
    name = name,
    sha256 = sha256,
    url = url,
) for name, sha256, origin, url in debs]

http_file(
    name = "kubernetes_cni",
    sha256 = "05ab3937bc68562e989dc143362ec4d4275262ba9f359338aed720fc914457a5",
    url = "https://storage.googleapis.com/kubernetes-release/network-plugins/cni-amd64-0799f5732f2a11b329d9e3d51b9c8f2e3759f2ff.tar.gz",
)

docker_pull(
    name = "debian-iptables-amd64",
    digest = "sha256:adde513f7b3561042cd2d2af4d2d355189bbb2f579584b5766e7d07be4f7e71e",  # v7
    registry = "gcr.io",
    repository = "google-containers/debian-iptables-amd64",
)
