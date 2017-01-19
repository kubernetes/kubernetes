package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)

config_setting(
    name = "embed_license_targets",
    values = {
        "define": "EMBED_LICENSE_TARGETS=true",
    },
    visibility = ["//visibility:private"],
)

pkg_tar(
    name = "kubernetes-src",
    extension = "tar.gz",
    files = [
        "//:all-srcs",
    ],
    package_dir = "kubernetes",
    strip_prefix = "//",
)

# FIXME: this should be configurable/auto-detected
PLATFORM_ARCH_STRING = "linux-amd64"

# KUBE_CLIENT_TARGETS
CLIENT_TARGETS = [
    "//cmd/kubectl",
    "//federation/cmd/kubefed",
]

# KUBE_NODE_TARGETS
NODE_TARGETS = [
    "//cmd/kube-proxy",
    "//cmd/kubelet",
]

# KUBE_SERVER_TARGETS
# No need to duplicate CLIENT_TARGETS or NODE_TARGETS here,
# since we include them in the actual build rule.
SERVER_TARGETS = [
    "//cmd/hyperkube",
    "//cmd/kube-aggregator",
    "//cmd/kube-apiserver",
    "//cmd/kube-controller-manager",
    "//cmd/kube-discovery",
    "//cmd/kubeadm",
    "//plugin/cmd/kube-scheduler",
]

# kube::golang::test_targets
TEST_BINARY_TARGETS = [
    "//cmd/gendocs",
    "//cmd/genkubedocs",
    "//cmd/genman",
    "//cmd/genswaggertypedocs",
    "//cmd/genyaml",
    "//cmd/linkcheck",
    "//cmd/mungedocs",
    "//examples/k8petstore/web-server/src",
    "//federation/cmd/genfeddocs",
    "//test/e2e:e2e.test",
    "//vendor:github.com/onsi/ginkgo/ginkgo_bin",
    "//cmd/kubemark",  # TODO: server platforms only
    "//test/e2e_node:e2e_node.test",  # TODO: server platforms only
]

TEST_PORTABLE_TARGETS = [
    "//federation/develop:all-srcs",
    "//hack:e2e.go",
    "//hack:federated-ginkgo-e2e.sh",
    "//hack:get-build.sh",
    "//hack:ginkgo-e2e.sh",
    "//hack/e2e-internal:all-srcs",
    "//hack/lib:all-srcs",
    "//test/e2e/testing-manifests:all-srcs",
    "//test/kubemark:all-srcs",
]

# Included in node and server tarballs.
LICENSE_TARGETS = [
    "//:Godeps/LICENSES",
    ":kubernetes-src.tar.gz",
]

pkg_tar(
    name = "_client-bin",
    files = CLIENT_TARGETS,
    mode = "0755",
    package_dir = "client/bin",
    visibility = ["//visibility:private"],
)

pkg_tar(
    name = "kubernetes-client-%s" % PLATFORM_ARCH_STRING,
    extension = "tar.gz",
    package_dir = "kubernetes",
    deps = [
        ":_client-bin",
    ],
)

pkg_tar(
    name = "_node-bin",
    files = NODE_TARGETS + CLIENT_TARGETS,
    mode = "0755",
    package_dir = "node/bin",
    visibility = ["//visibility:private"],
)

pkg_tar(
    name = "kubernetes-node-%s" % PLATFORM_ARCH_STRING,
    extension = "tar.gz",
    files = select({
        ":embed_license_targets": LICENSE_TARGETS,
        "//conditions:default": [],
    }),
    mode = "0644",
    package_dir = "kubernetes",
    deps = [
        ":_node-bin",
    ],
)

pkg_tar(
    name = "_server-bin",
    files = SERVER_TARGETS + NODE_TARGETS + CLIENT_TARGETS + [
        "//build:docker-artifacts",
    ],
    mode = "0755",
    package_dir = "server/bin",
    visibility = ["//visibility:private"],
)

genrule(
    name = "dummy",
    outs = [".dummy"],
    cmd = "touch $@",
)

# Some of the startup scripts fail if there isn't an addons/ directory in the server tarball.
pkg_tar(
    name = "_server-addons",
    files = [
        ":.dummy",
    ],
    package_dir = "addons",
    visibility = ["//visibility:private"],
)

pkg_tar(
    name = "kubernetes-server-%s" % PLATFORM_ARCH_STRING,
    extension = "tar.gz",
    files = select({
        ":embed_license_targets": LICENSE_TARGETS,
        "//conditions:default": [],
    }),
    mode = "0644",
    package_dir = "kubernetes",
    deps = [
        ":_server-addons",
        ":_server-bin",
    ],
)

pkg_tar(
    name = "_test-bin",
    files = TEST_BINARY_TARGETS,
    mode = "0755",
    package_dir = "platforms/" + PLATFORM_ARCH_STRING.replace("-", "/"),
    # TODO: how to make this multiplatform?
    visibility = ["//visibility:private"],
)

pkg_tar(
    name = "kubernetes-test",
    extension = "tar.gz",
    files = TEST_PORTABLE_TARGETS,
    package_dir = "kubernetes",
    strip_prefix = "//",
    deps = [
        # TODO: how to make this multiplatform?
        ":_test-bin",
    ],
)

pkg_tar(
    name = "_full_server",
    files = [
        ":kubernetes-manifests.tar.gz",
        ":kubernetes-salt.tar.gz",
    ],
    package_dir = "server",
    visibility = ["//visibility:private"],
)

pkg_tar(
    name = "kubernetes",
    extension = "tar.gz",
    files = [
        # TODO: the version file
        "//:README.md",
        "//:Vagrantfile",
        "//cluster:all-srcs",
        "//docs:all-srcs",
        "//examples:all-srcs",
        "//third_party/htpasswd:all-srcs",
    ] + select({
        ":embed_license_targets": ["//:Godeps/LICENSES"],
        "//conditions:default": [],
    }),
    package_dir = "kubernetes",
    strip_prefix = "//",
    deps = [
        ":_full_server",
        "//federation:release",
    ],
)

pkg_tar(
    name = "kubernetes-manifests",
    extension = "tar.gz",
    deps = [
        "//cluster:manifests",
    ],
)

pkg_tar(
    name = "kubernetes-salt",
    extension = "tar.gz",
    deps = [
        "//cluster/saltbase:salt",
    ],
)

filegroup(
    name = "release-tars",
    srcs = [
        ":kubernetes.tar.gz",
        ":kubernetes-client-%s.tar.gz" % PLATFORM_ARCH_STRING,
        ":kubernetes-node-%s.tar.gz" % PLATFORM_ARCH_STRING,
        ":kubernetes-server-%s.tar.gz" % PLATFORM_ARCH_STRING,
        ":kubernetes-manifests.tar.gz",
        ":kubernetes-salt.tar.gz",
        ":kubernetes-src.tar.gz",
        ":kubernetes-test.tar.gz",
    ],
)
