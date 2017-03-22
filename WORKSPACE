git_repository(
    name = "io_bazel_rules_go",
    commit = "7828452850597b52b49ec603b23f8ad2bcb22aed",
    remote = "https://github.com/bazelbuild/rules_go.git",
)

git_repository(
    name = "io_kubernetes_build",
    commit = "685f15b90b454af3086ab071fdea1b6db213d1fb",
    remote = "https://github.com/kubernetes/repo-infra.git",
)

git_repository(
    name = "io_bazel",
    commit = "3b29803eb528ff525c7024190ffbf4b08c598cf2",
    remote = "https://github.com/ixdy/bazel.git",
)

load("@io_bazel_rules_go//go:def.bzl", "go_repositories")

go_repositories()

# for building docker base images
debs = (
    (
        "busybox_deb",
        "f262cc9cf893740bb70c3dd01da9429b858c94be696badd4a702e0a8c7f6f80b",
        "http://ftp.us.debian.org/debian/pool/main/b/busybox/busybox-static_1.22.0-19+b1_amd64.deb",
    ),
    (
        "libc_deb",
        "6bbd506b171a9f29b09fde77e2749c0aa0c1439058df9d1a6408d464069b7dd6",
        "http://ftp.us.debian.org/debian/pool/main/g/glibc/libc6_2.24-9_amd64.deb",
    ),
    (
        "iptables_deb",
        "7747388a97ba71fede302d70361c81d486770a2024185514c18b5d8eab6aaf4e",
        "http://ftp.us.debian.org/debian/pool/main/i/iptables/iptables_1.4.21-2+b1_amd64.deb",
    ),
    (
        "libnetlink_deb",
        "5d486022cd9e047e9afbb1617cf4519c0decfc3d2c1fad7e7fe5604943dbbf37",
        "http://ftp.us.debian.org/debian/pool/main/libn/libnfnetlink/libnfnetlink0_1.0.1-3_amd64.deb",
    ),
    (
        "libxtables_deb",
        "6783f316af4cbf3ada8b9a2b7bb5f53a87c0c2575c1903ce371fdbd45d3626c6",
        "http://ftp.us.debian.org/debian/pool/main/i/iptables/libxtables10_1.4.21-2+b1_amd64.deb",
    ),
    (
        "iproute2_deb",
        "3ce9cb1d03a2a1359cbdd4f863b15d0c906096bf713e8eb688149da2f4e350bc",
        "http://ftp.us.debian.org/debian/pool/main/i/iproute2/iproute_3.16.0-2_all.deb",
    ),
)

[http_file(
    name = name,
    sha256 = sha256,
    url = url,
) for name, sha256, url in debs]

http_file(
    name = "kubernetes_cni",
    sha256 = "05ab3937bc68562e989dc143362ec4d4275262ba9f359338aed720fc914457a5",
    url = "https://storage.googleapis.com/kubernetes-release/network-plugins/cni-amd64-0799f5732f2a11b329d9e3d51b9c8f2e3759f2ff.tar.gz",
)
