package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_docker//docker:docker.bzl", "docker_build", "docker_bundle")
load("@io_bazel_rules_docker//container:container.bzl", "container_image")
load("@io_kubernetes_build//defs:build.bzl", "release_filegroup")

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
        "//build/rpms:all-srcs",
        "//build/visible_to:all-srcs",
    ],
    tags = ["automanaged"],
)

# ensure /etc/nsswitch.conf exists so go's resolver respects /etc/hosts
container_image(
    name = "busybox-with-nsswitch",
    base = "@official_busybox//image",
    directory = "/etc",
    files = ["nsswitch.conf"],
    mode = "0644",
)

# This list should roughly match kube::build::get_docker_wrapped_binaries()
# in build/common.sh.
DOCKERIZED_BINARIES = {
    "cloud-controller-manager": {
        "base": ":busybox-with-nsswitch",
        "target": "//cmd/cloud-controller-manager:cloud-controller-manager",
    },
    "kube-apiserver": {
        "base": ":busybox-with-nsswitch",
        "target": "//cmd/kube-apiserver:kube-apiserver",
    },
    "kube-controller-manager": {
        "base": ":busybox-with-nsswitch",
        "target": "//cmd/kube-controller-manager:kube-controller-manager",
    },
    "kube-scheduler": {
        "base": ":busybox-with-nsswitch",
        "target": "//cmd/kube-scheduler:kube-scheduler",
    },
    "kube-proxy": {
        "base": "@debian-iptables-amd64//image",
        "target": "//cmd/kube-proxy:kube-proxy",
    },
}

[docker_build(
    name = binary + "-internal",
    base = meta["base"],
    cmd = ["/usr/bin/" + binary],
    debs = [
        "//build/debs:%s.deb" % binary,
    ],
    symlinks = {
        # Some cluster startup scripts expect to find the binaries in /usr/local/bin,
        # but the debs install the binaries into /usr/bin.
        "/usr/local/bin/" + binary: "/usr/bin/" + binary,
    },
) for binary, meta in DOCKERIZED_BINARIES.items()]

[docker_bundle(
    name = binary,
    # TODO(thockin): remove the google_containers name after release 1.11.
    images = {
        "k8s.gcr.io/%s:{STABLE_DOCKER_TAG}" % binary: binary + "-internal",
        "gcr.io/google_containers/%s:{STABLE_DOCKER_TAG}" % binary: binary + "-internal",
    },
    stamp = True,
) for binary in DOCKERIZED_BINARIES.keys()]

[genrule(
    name = binary + "_docker_tag",
    srcs = [meta["target"]],
    outs = [binary + ".docker_tag"],
    cmd = "grep ^STABLE_DOCKER_TAG bazel-out/stable-status.txt | awk '{print $$2}' >$@",
    stamp = 1,
) for binary, meta in DOCKERIZED_BINARIES.items()]

genrule(
    name = "os_package_version",
    outs = ["version"],
    cmd = """
grep ^STABLE_BUILD_SCM_REVISION bazel-out/stable-status.txt \
    | awk '{print $$2}' \
    | sed -e 's/^v//' -Ee 's/-([a-z]+)/~\\1/' -e 's/-/+/g' \
    >$@
""",
    stamp = 1,
)

genrule(
    name = "cni_package_version",
    outs = ["cni_version"],
    cmd = "echo 0.5.1 >$@",
)

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
    ],
)

# KUBE_NODE_TARGETS
release_filegroup(
    name = "node-targets",
    srcs = [
        "//cmd/kube-proxy",
        "//cmd/kubeadm",
        "//cmd/kubelet",
    ],
)

# KUBE_SERVER_TARGETS
# No need to duplicate CLIENT_TARGETS or NODE_TARGETS here,
# since we include them in the actual build rule.
release_filegroup(
    name = "server-targets",
    srcs = [
        "//cluster/gce/gci/mounter",
        "//cmd/cloud-controller-manager",
        "//cmd/hyperkube",
        "//cmd/kube-apiserver",
        "//cmd/kube-controller-manager",
        "//cmd/kube-scheduler",
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
        "//test/e2e:e2e.test",
        "//test/e2e_node:e2e_node.test",  # TODO: server platforms only
        "//vendor/github.com/onsi/ginkgo/ginkgo",
    ],
)

# KUBE_TEST_PORTABLE
filegroup(
    name = "test-portable-targets",
    srcs = [
        "//hack:e2e.go",
        "//hack:get-build.sh",
        "//hack:ginkgo-e2e.sh",
        "//hack/e2e-internal:all-srcs",
        "//hack/lib:all-srcs",
        "//test/e2e/testing-manifests:all-srcs",
        "//test/kubemark:all-srcs",
    ],
)
