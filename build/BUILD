package(default_visibility = ["//visibility:public"])

load("@io_k8s_repo_infra//defs:build.bzl", "release_filegroup")
load("@io_k8s_repo_infra//defs:pkg.bzl", "pkg_tar")
load(":code_generation_test.bzl", "code_generation_test_suite")
load(":container.bzl", "multi_arch_container", "multi_arch_container_push")
load(":platforms.bzl", "SERVER_PLATFORMS", "for_platforms")

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
        "//build/go-runner:all-srcs",
        "//build/release-tars:all-srcs",
        "//build/visible_to:all-srcs",
    ],
    tags = ["automanaged"],
)

# This list should roughly match kube::build::get_docker_wrapped_binaries()
# in build/common.sh.
DOCKERIZED_BINARIES = {
    "kube-apiserver": {
        "base": "@go-runner-linux-{ARCH}//image",
        "target": "//cmd/kube-apiserver:kube-apiserver",
    },
    "kube-controller-manager": {
        "base": "@go-runner-linux-{ARCH}//image",
        "target": "//cmd/kube-controller-manager:kube-controller-manager",
    },
    "kube-scheduler": {
        "base": "@go-runner-linux-{ARCH}//image",
        "target": "//cmd/kube-scheduler:kube-scheduler",
    },
    "kube-proxy": {
        "base": "@debian-iptables-{ARCH}//image",
        "target": "//cmd/kube-proxy:kube-proxy",
    },
}

[pkg_tar(
    name = "%s-data-%s.tar" % (binary, arch),
    srcs = select({"@io_bazel_rules_go//go/platform:" + arch: ["//cmd/" + binary]}),
    mode = "0755",
    package_dir = "/usr/bin",
    tags = ["manual"],
    visibility = ["//visibility:private"],
) for binary in DOCKERIZED_BINARIES.keys() for arch in SERVER_PLATFORMS["linux"]]

# When pushing to gcr.io, we want to use an arch, since the archless name is now used for a
# manifest list. Bazel doesn't support manifest lists (yet), so we can't do that either.
[multi_arch_container(
    name = binary,
    architectures = SERVER_PLATFORMS["linux"],
    base = meta["base"],
    cmd = ["/usr/bin/" + binary],
    # Since the multi_arch_container macro replaces the {ARCH} format string,
    # we need to escape the stamping vars.
    docker_push_tags = ["{{STABLE_DOCKER_PUSH_REGISTRY}}/%s-{ARCH}:{{STABLE_DOCKER_TAG}}" % binary],
    docker_tags = ["{{STABLE_DOCKER_REGISTRY}}/%s-{ARCH}:{{STABLE_DOCKER_TAG}}" % binary],
    stamp = True,
    symlinks = {
        # Some cluster startup scripts expect to find the binaries in /usr/local/bin,
        # but the debs install the binaries into /usr/bin.
        "/usr/local/bin/" + binary: "/usr/bin/" + binary,
    },
    tags = ["manual"],
    tars = select(for_platforms(
        for_server = [":%s-data-{ARCH}.tar" % binary],
        only_os = "linux",
    )),
    visibility = ["//visibility:private"],
) for binary, meta in DOCKERIZED_BINARIES.items()]

# Also roll up all images into a single bundle to push with one target.
multi_arch_container_push(
    name = "server-images",
    architectures = SERVER_PLATFORMS["linux"],
    docker_tags_images = {
        "{{STABLE_DOCKER_PUSH_REGISTRY}}/%s-{ARCH}:{{STABLE_DOCKER_TAG}}" % binary: "%s-internal" % binary
        for binary in DOCKERIZED_BINARIES.keys()
    },
    tags = ["manual"],
)

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
            "//cluster/images/conformance/go-runner",
        ],
    )),
)

# KUBE_TEST_PORTABLE
filegroup(
    name = "test-portable-targets",
    srcs = [
        "//hack:get-build.sh",
        "//hack:ginkgo-e2e.sh",
        "//hack/e2e-internal:all-srcs",
        "//hack/lib:all-srcs",
        "//test/e2e/testing-manifests:all-srcs",
        "//test/kubemark:all-srcs",
    ],
)
