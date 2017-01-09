package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_defs/docker:docker.bzl", "docker_build")

docker_build(
    name = "busybox",
    debs = [
        "@busybox_deb//file",
    ],
    symlinks = {
        "/bin/sh": "/bin/busybox",
        "/usr/bin/busybox": "/bin/busybox",
        "/usr/sbin/busybox": "/bin/busybox",
        "/sbin/busybox": "/bin/busybox",
    },
)

docker_build(
    name = "busybox-libc",
    base = ":busybox",
    debs = [
        "@libc_deb//file",
    ],
)

docker_build(
    name = "busybox-net",
    base = ":busybox-libc",
    debs = [
        "@iptables_deb//file",
        "@iproute2_deb//file",
        "@libnetlink_deb//file",
        "@libxtables_deb//file",
    ],
)

[docker_build(
    name = binary,
    base = ":busybox-libc",
    cmd = ["/usr/bin/" + binary],
    debs = [
        "//build/debs:%s.deb" % binary,
    ],
    repository = "gcr.io/google-containers",
) for binary in [
    "kube-apiserver",
    "kube-controller-manager",
    "kube-scheduler",
    "kube-aggregator",
]]

docker_build(
    name = "kube-proxy",
    base = ":busybox-net",
    cmd = ["/usr/bin/kube-proxy"],
    debs = [
        "//build/debs:kube-proxy.deb",
    ],
    repository = "gcr.io/google-containers",
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//build/debs:all-srcs",
    ],
    tags = ["automanaged"],
)
