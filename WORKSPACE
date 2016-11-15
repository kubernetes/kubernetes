git_repository(
    name = "io_bazel_rules_go",
    commit = "e0b19317b39357823b26c7e266596c8066e8f8e6",
    remote = "https://github.com/bazelbuild/rules_go.git",
)

load("@io_bazel_rules_go//go:def.bzl", "go_repositories")

go_repositories()

# for building docker base images
debs = (
    (
        "busybox_deb",
        "51651980a993b02c8dc663a5539a4d83704e56c2fed93dd8d1b2580e61319af5",
        "http://ftp.us.debian.org/debian/pool/main/b/busybox/busybox-static_1.22.0-19_amd64.deb",
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
