http_archive(
    name = "io_bazel_rules_go",
    sha256 = "a1cae429e9d591017421150e3173478c46c693bc594322c7fa7e6cb5f672ef59",
    strip_prefix = "rules_go-805fd1566500997379806373feb05e138a4dfe28",
    urls = ["https://github.com/bazelbuild/rules_go/archive/805fd1566500997379806373feb05e138a4dfe28.tar.gz"],
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
load("@io_bazel_rules_docker//docker:docker.bzl", "docker_repositories")

go_repositories(
    go_version = "1.8.1",
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
    (
        "libc_deb",
        "372aac4a9ce9dbb26a08de0b9c41b0500ba019430295d29f39566483f5f32732",
        "http://ftp.us.debian.org/debian/pool/main/g/glibc/libc6_2.24-10_amd64.deb",
        "https://storage.googleapis.com/kubernetes-release/debs/libc6_2.24-10_amd64.deb",
    ),
    (
        "iptables_deb",
        "7747388a97ba71fede302d70361c81d486770a2024185514c18b5d8eab6aaf4e",
        "http://ftp.us.debian.org/debian/pool/main/i/iptables/iptables_1.4.21-2+b1_amd64.deb",
        "https://storage.googleapis.com/kubernetes-release/debs/iptables_1.4.21-2+b1_amd64.deb",
    ),
    (
        "libnetlink_deb",
        "5d486022cd9e047e9afbb1617cf4519c0decfc3d2c1fad7e7fe5604943dbbf37",
        "http://ftp.us.debian.org/debian/pool/main/libn/libnfnetlink/libnfnetlink0_1.0.1-3_amd64.deb",
        "https://storage.googleapis.com/kubernetes-release/debs/libnfnetlink0_1.0.1-3_amd64.deb",
    ),
    (
        "libxtables_deb",
        "6783f316af4cbf3ada8b9a2b7bb5f53a87c0c2575c1903ce371fdbd45d3626c6",
        "http://ftp.us.debian.org/debian/pool/main/i/iptables/libxtables10_1.4.21-2+b1_amd64.deb",
        "https://storage.googleapis.com/kubernetes-release/debs/libxtables10_1.4.21-2+b1_amd64.deb",
    ),
    (
        "iproute2_deb",
        "3ce9cb1d03a2a1359cbdd4f863b15d0c906096bf713e8eb688149da2f4e350bc",
        "http://ftp.us.debian.org/debian/pool/main/i/iproute2/iproute_3.16.0-2_all.deb",
        "https://storage.googleapis.com/kubernetes-release/debs/iproute_3.16.0-2_all.deb",
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
