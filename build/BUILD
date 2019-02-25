package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_docker//container:container.bzl", "container_bundle", "container_image")
load("@io_k8s_repo_infra//defs:build.bzl", "release_filegroup")
load(":code_generation_test.bzl", "code_generation_test_suite")
load(":platforms.bzl", "for_platforms")

code_generation_test_suite(
    name = "code_generation_tests",
)

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

# This list should roughly match kube::build::get_docker_wrapped_binaries()
# in build/common.sh.
DOCKERIZED_BINARIES = {
    "cloud-controller-manager": {
        "base": select(for_platforms(
            for_node = "@debian-base-{ARCH}//image",
            only_os = "linux",
        )),
        "target": "//cmd/cloud-controller-manager:cloud-controller-manager",
    },
    "kube-apiserver": {
        "base": select(for_platforms(
            for_node = "@debian-base-{ARCH}//image",
            only_os = "linux",
        )),
        "target": "//cmd/kube-apiserver:kube-apiserver",
    },
    "kube-controller-manager": {
        "base": select(for_platforms(
            for_node = "@debian-base-{ARCH}//image",
            only_os = "linux",
        )),
        "target": "//cmd/kube-controller-manager:kube-controller-manager",
    },
    "kube-scheduler": {
        "base": select(for_platforms(
            for_node = "@debian-base-{ARCH}//image",
            only_os = "linux",
        )),
        "target": "//cmd/kube-scheduler:kube-scheduler",
    },
    "kube-proxy": {
        "base": select(for_platforms(
            for_node = "@debian-iptables-{ARCH}//image",
            only_os = "linux",
        )),
        "target": "//cmd/kube-proxy:kube-proxy",
    },
}

[container_image(
    name = binary + "-internal",
    base = meta["base"],
    cmd = ["/usr/bin/" + binary],
    debs = select(for_platforms(
        for_node = ["//build/debs:%s-{ARCH}.deb" % binary],
        only_os = "linux",
    )),
    stamp = True,
    symlinks = {
        # Some cluster startup scripts expect to find the binaries in /usr/local/bin,
        # but the debs install the binaries into /usr/bin.
        "/usr/local/bin/" + binary: "/usr/bin/" + binary,
    },
) for binary, meta in DOCKERIZED_BINARIES.items()]

[container_bundle(
    name = binary,
    images = {
        "k8s.gcr.io/%s:{STABLE_DOCKER_TAG}" % binary: binary + "-internal",
    },
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

release_filegroup(
    name = "docker-artifacts",
    srcs = [":%s.tar" % binary for binary in DOCKERIZED_BINARIES.keys()] +
           [":%s.docker_tag" % binary for binary in DOCKERIZED_BINARIES.keys()],
)

# KUBE_CLIENT_TARGETS
release_filegroup(
    name = "client-targets",
    conditioned_srcs = for_platforms(for_client = [
        "//cmd/kubectl",
    ]),
)

# KUBE_NODE_TARGETS
release_filegroup(
    name = "node-targets",
    conditioned_srcs = for_platforms(for_node = [
        "//cmd/kube-proxy",
        "//cmd/kubeadm",
        "//cmd/kubelet",
    ]),
)

# KUBE_SERVER_TARGETS
# No need to duplicate CLIENT_TARGETS or NODE_TARGETS here,
# since we include them in the actual build rule.
release_filegroup(
    name = "server-targets",
    conditioned_srcs = for_platforms(for_server = [
        "//cluster/gce/gci/mounter",
        "//cmd/cloud-controller-manager",
        "//cmd/hyperkube",
        "//cmd/kube-apiserver",
        "//cmd/kube-controller-manager",
        "//cmd/kube-scheduler",
    ]),
)

# kube::golang::test_targets
filegroup(
    name = "test-targets",
    srcs = select(for_platforms(
        for_server = [
            "//cmd/kubemark",
            "//test/e2e_node:e2e_node.test_binary",
        ],
        for_test = [
            "//cmd/gendocs",
            "//cmd/genkubedocs",
            "//cmd/genman",
            "//cmd/genswaggertypedocs",
            "//cmd/genyaml",
            "//cmd/linkcheck",
            "//test/e2e:e2e.test_binary",
            "//vendor/github.com/onsi/ginkgo/ginkgo",
        ],
    )),
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
