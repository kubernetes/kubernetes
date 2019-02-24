package(default_visibility = ["//visibility:public"])

load("@io_bazel_rules_docker//container:container.bzl", "container_bundle", "container_image")
load("@io_k8s_repo_infra//defs:build.bzl", "release_filegroup")
load(":code_generation_test.bzl", "code_generation_test_suite")
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

[[container_image(
    name = "%s-internal-%s" % (binary, arch),
    base = meta["base"],
    cmd = ["/usr/bin/" + binary],
    debs = ["//build/debs:%s-%s.deb" % (binary, arch)],
    stamp = True,
    symlinks = {
        # Some cluster startup scripts expect to find the binaries in /usr/local/bin,
        # but the debs install the binaries into /usr/bin.
        "/usr/local/bin/" + binary: "/usr/bin/" + binary,
    },
    visibility = ["//visibility:private"],
) for binary, meta in DOCKERIZED_BINARIES.items()] for arch in SERVER_PLATFORMS["linux"]]

# Also create aliases for the arch-specific images defined above.
# The alias doesn't create a new file (like a genrule would);
# instead, we are using it with a select() conditional to
# be able to refer to the correct architecture in a consistent way.
# (Notably, container_bundle does not seem to support using a select()
# in the images attribute, so we defer that selection to this rule.)
[alias(
    name = "%s-internal" % binary,
    actual = select(for_platforms(
        for_server = ":%s-internal-{ARCH}" % binary,
        only_os = "linux",
    )),
    visibility = ["//visibility:private"],
) for binary in DOCKERIZED_BINARIES.keys()]

# We create two container bundles with the desired tags; one with an architecture in the name
# and one without.
# In the bash-based build (build/lib/release.sh), the images built for amd64 do not use
# an arch in their name (but other arches do), and the GCE cluster scripts
# (which sideload the images via tarfiles) expect there not to be an arch.
# When pushing to gcr.io, we want to use an arch, since the archless name is now used for a
# manifest list. Bazel doesn't support manifest lists (yet), so we can't do that either.
# Instead, for now, we use the archless name for the image tars saved in the server tarball,
# to satisfy GCE and other similar providers. (If one were to pull the images via the manifest
# list, the arch wouldn't appear in the name either.)
# The bundle with the arch isn't used currently, but might be at some point for pushing to gcr.io
# and to then create a manifest list (possibly outside of bazel).
[[container_bundle(
    name = "%s-%s" % (binary, arch),
    images = {
        "k8s.gcr.io/%s-%s:{STABLE_DOCKER_TAG}" % (binary, arch): "%s-internal-%s" % (binary, arch),
    },
    tags = ["manual"],
    visibility = ["//visibility:public"],
) for binary in DOCKERIZED_BINARIES.keys()] for arch in SERVER_PLATFORMS["linux"]]

[container_bundle(
    name = binary,
    images = {
        "k8s.gcr.io/%s:{STABLE_DOCKER_TAG}" % binary: binary + "-internal",
    },
    tags = ["manual"],
    visibility = ["//visibility:private"],
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
