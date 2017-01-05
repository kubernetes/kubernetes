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
    "kubernetes-discovery",
]]

docker_build(
    name = "kube-proxy",
    base = ":busybox-net",
    cmd = ["/kube-proxy"],
    debs = [
        "//build/debs:kube-proxy.deb",
    ],
    repository = "gcr.io/google-containers",
)
