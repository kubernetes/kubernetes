package(default_visibility = ["//visibility:public"])

load("@io_bazel//tools/build_defs/docker:docker.bzl", "docker_build")
load("@io_kubernetes_build//defs:build.bzl", "md5sum", "release_filegroup")

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
    name = "busybox-net",
    base = ":busybox",
    debs = [
        "@libc_deb//file",
        "@iptables_deb//file",
        "@iproute2_deb//file",
        "@libnetlink_deb//file",
        "@libxtables_deb//file",
    ],
)

DOCKERIZED_BINARIES = {
    "kube-apiserver": {
        "base": ":busybox",
        "target": "//cmd/kube-apiserver:kube-apiserver",
    },
    "kube-controller-manager": {
        "base": ":busybox",
        "target": "//cmd/kube-controller-manager:kube-controller-manager",
    },
    "kube-scheduler": {
        "base": ":busybox",
        "target": "//plugin/cmd/kube-scheduler:kube-scheduler",
    },
    "kube-proxy": {
        "base": ":busybox-net",
        "target": "//cmd/kube-proxy:kube-proxy",
    },
}

[md5sum(
    name = binary + ".docker_tag",
    src = meta["target"],
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

release_filegroup(
    name = "docker-artifacts",
    srcs = [":%s.tar" % binary for binary in DOCKERIZED_BINARIES.keys()] +
           [":%s.docker_tag" % binary for binary in DOCKERIZED_BINARIES.keys()],
)

# KUBE_CLIENT_TARGETS
release_filegroup(
    name = "client-targets",
    srcs = [
        "//cmd/kubectl",
        "//federation/cmd/kubefed",
    ],
)

# KUBE_NODE_TARGETS
release_filegroup(
    name = "node-targets",
    srcs = [
        "//cmd/kube-proxy",
        "//cmd/kubelet",
    ],
)

# KUBE_SERVER_TARGETS
# No need to duplicate CLIENT_TARGETS or NODE_TARGETS here,
# since we include them in the actual build rule.
release_filegroup(
    name = "server-targets",
    srcs = [
        "//cmd/cloud-controller-manager",
        "//cmd/hyperkube",
        "//cmd/kube-apiserver",
        "//cmd/kube-controller-manager",
        "//cmd/kubeadm",
        "//plugin/cmd/kube-scheduler",
        "//vendor/k8s.io/kube-aggregator",
    ],
)

# kube::golang::test_targets
filegroup(
    name = "test-targets",
    srcs = [
        "//cmd/gendocs",
        "//cmd/genkubedocs",
        "//cmd/genman",
        "//cmd/genswaggertypedocs",
        "//cmd/genyaml",
        "//cmd/kubemark",  # TODO: server platforms only
        "//cmd/linkcheck",
        "//cmd/mungedocs",
        "//federation/cmd/genfeddocs",
        "//test/e2e:e2e.test",
        "//test/e2e_node:e2e_node.test",  # TODO: server platforms only
        "//vendor/github.com/onsi/ginkgo/ginkgo",
    ],
)

# KUBE_TEST_PORTABLE
filegroup(
    name = "test-portable-targets",
    srcs = [
        "//federation/develop:all-srcs",
        "//hack:e2e.go",
        "//hack:federated-ginkgo-e2e.sh",
        "//hack:get-build.sh",
        "//hack:ginkgo-e2e.sh",
        "//hack/e2e-internal:all-srcs",
        "//hack/lib:all-srcs",
        "//test/e2e/testing-manifests:all-srcs",
        "//test/kubemark:all-srcs",
    ],
)
