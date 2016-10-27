package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_defs/docker:docker.bzl", "docker_build")

server_binaries = {
    "kube-apiserver": "//cmd/kube-apiserver",
    "kube-controller-manager": "//cmd/kube-controller-manager",
    "kube-scheduler": "//plugin/cmd/kube-scheduler",
}

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
    name = name,
    base = ":busybox-libc",
    cmd = ["/" + name],
    files = [
        label,
    ],
    repository = "gcr.io/google-containers",
) for name, label in server_binaries.items()]

docker_build(
    name = "kube-proxy",
    base = ":busybox-net",
    cmd = ["/kube-proxy"],
    files = [
        "//cmd/kube-proxy",
    ],
    repository = "gcr.io/google-containers",
)
