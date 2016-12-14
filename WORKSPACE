git_repository(
    name = "io_bazel_rules_go",
    commit = "d0142854a22a0dd98306280e897e64086289a0de",
    remote = "https://github.com/bazelbuild/rules_go.git",
)

git_repository(
    name = "io_kubernetes_build",
    commit = "418b8e976cb32d94fd765c80f2b04e660c5ec4ec",
    remote = "https://github.com/kubernetes/release.git",
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
        "ee4d9dea08728e2c2bbf43d819c3c7e61798245fab4b983ae910865980f791ad",
        "http://ftp.us.debian.org/debian/pool/main/g/glibc/libc6_2.19-18+deb8u6_amd64.deb",
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
    sha256 = "ddcb7a429f82b284a13bdb36313eeffd997753b6fa5191205f1e978dcfeb0792",
    url = " https://storage.googleapis.com/kubernetes-release/network-plugins/cni-amd64-07a8a28637e97b22eb8dfe710eeae1344f69d16e.tar.gz",
)
