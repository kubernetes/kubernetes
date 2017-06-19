http_archive(
    name = "io_bazel_rules_go",
    sha256 = "a1596c14c799d5a1b5f49ca28fa948414c2242110d69ef324d6ed160ec890dbf",
    strip_prefix = "rules_go-03c634753160632c00f506afeafc819fbea4c422",
    urls = ["https://github.com/bazelbuild/rules_go/archive/03c634753160632c00f506afeafc819fbea4c422.tar.gz"],
)

http_archive(
    name = "io_kubernetes_build",
    sha256 = "a9fb7027f060b868cdbd235a0de0971b557b9d26f9c89e422feb80f48d8c0e90",
    strip_prefix = "repo-infra-9dedd5f4093884c133ad5ea73695b28338b954ab",
    urls = ["https://github.com/kubernetes/repo-infra/archive/9dedd5f4093884c133ad5ea73695b28338b954ab.tar.gz"],
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
    digest = "sha256:bc20977ac38abfb43071b4c61c4b7edb30af894c05eb06758dd61d05118d2842",  # v7
    registry = "gcr.io",
    repository = "google-containers/debian-iptables-amd64",
)
