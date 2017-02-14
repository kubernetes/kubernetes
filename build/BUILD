package(default_visibility = ["//visibility:public"])

load("@io_bazel//tools/build_defs/docker:docker.bzl", "docker_build")

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//build/debs:all-srcs",
        "//build/release-tars:all-srcs",
    ],
    tags = ["automanaged"],
)

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

DOCKERIZED_BINARIES = {
    "kube-apiserver": {
        "base": ":busybox-libc",
        "target": "//cmd/kube-apiserver:kube-apiserver",
    },
    "kube-controller-manager": {
        "base": ":busybox-libc",
        "target": "//cmd/kube-controller-manager:kube-controller-manager",
    },
    "kube-scheduler": {
        "base": ":busybox-libc",
        "target": "//plugin/cmd/kube-scheduler:kube-scheduler",
    },
    "kube-proxy": {
        "base": ":busybox-net",
        "target": "//cmd/kube-proxy:kube-proxy",
    },
}

[genrule(
    name = binary + "_docker_tag",
    srcs = [meta["target"]],
    outs = [binary + ".docker_tag"],
    # Currently each target has two outputs, the binary and its library, so hash only the first item (the binary).
    # This can be made less hacky when we have static linking working.
    cmd = "md5sum $(locations " + meta["target"] + ") | grep '" + binary + "'$$ | cut -f1 -d' ' | tr -d '\n' > $@",
) for binary, meta in DOCKERIZED_BINARIES.items()]

[docker_build(
    name = binary,
    base = meta["base"],
    cmd = ["/usr/bin/" + binary],
    debs = [
        "//build/debs:%s.deb" % binary,
    ],
    image_tags = [
        "@%s.docker_tag" % binary,
    ],
    repository = "gcr.io/google_containers/" + binary,
    repository_append_package = False,
    symlinks = {
        # Some cluster startup scripts expect to find the binaries in /usr/local/bin,
        # but the debs install the binaries into /usr/bin.
        "/usr/local/bin/" + binary: "/usr/bin/" + binary,
    },
) for binary, meta in DOCKERIZED_BINARIES.items()]

filegroup(
    name = "docker-artifacts",
    srcs = [":%s.tar" % binary for binary in DOCKERIZED_BINARIES.keys()] +
           [":%s.docker_tag" % binary for binary in DOCKERIZED_BINARIES.keys()],
)
