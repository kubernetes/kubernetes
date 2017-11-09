package(default_visibility = ["//visibility:public"])

load("@io_bazel//tools/build_defs/pkg:pkg.bzl", "pkg_tar")
load("@io_kubernetes_build//defs:build.bzl", "release_filegroup")

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
    name = "package_src",
    values = {
        "define": "PACKAGE_SRC=true",
    },
    visibility = ["//visibility:private"],
)

genrule(
    name = "kubernetes-src-readme",
    outs = ["README-src.txt"],
    cmd = """
echo For build efficiency, the src was not included in this release.>$@
echo The full source code can be viewed at >>$@
echo -n https://github.com/kubernetes/kubernetes/tree/ >>$@
grep ^STABLE_BUILD_GIT_COMMIT bazel-out/stable-status.txt | cut -d' ' -f2 >>$@
""",
    stamp = 1,
)

pkg_tar(
    name = "kubernetes-src",
    extension = "tar.gz",
    files = select({
        ":package_src": ["//:all-srcs"],
        "//conditions:default": ["README-src.txt"],
    }),
    package_dir = "kubernetes",
    strip_prefix = select({
        ":package_src": "//",
        "//conditions:default": ".",
    }),
)

# FIXME: this should be configurable/auto-detected
PLATFORM_ARCH_STRING = "linux-amd64"

# Included in node and server tarballs.
filegroup(
    name = "license-targets",
    srcs = [
        ":kubernetes-src.tar.gz",
        "//:Godeps/LICENSES",
    ],
    visibility = ["//visibility:private"],
)

pkg_tar(
    name = "_client-bin",
    files = ["//build:client-targets"],
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
    files = [
        "//build:client-targets",
        "//build:node-targets",
    ],
    mode = "0755",
    package_dir = "node/bin",
    visibility = ["//visibility:private"],
)

pkg_tar(
    name = "kubernetes-node-%s" % PLATFORM_ARCH_STRING,
    extension = "tar.gz",
    files = [":license-targets"],
    mode = "0644",
    package_dir = "kubernetes",
    deps = [
        ":_node-bin",
    ],
)

pkg_tar(
    name = "_server-bin",
    files = [
        "//build:client-targets",
        "//build:docker-artifacts",
        "//build:node-targets",
        "//build:server-targets",
    ],
    mode = "0755",
    package_dir = "server/bin",
    visibility = ["//visibility:private"],
)

genrule(
    name = "dummy",
    outs = [".dummy"],
    cmd = "touch $@",
    visibility = ["//visibility:private"],
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
    files = [":license-targets"],
    mode = "0644",
    package_dir = "kubernetes",
    deps = [
        ":_server-addons",
        ":_server-bin",
    ],
)

pkg_tar(
    name = "_test-bin",
    files = ["//build:test-targets"],
    mode = "0755",
    package_dir = "platforms/" + PLATFORM_ARCH_STRING.replace("-", "/"),
    # TODO: how to make this multiplatform?
    visibility = ["//visibility:private"],
)

pkg_tar(
    name = "kubernetes-test",
    extension = "tar.gz",
    files = ["//build:test-portable-targets"],
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
        "//:Godeps/LICENSES",
        "//:README.md",
        "//:Vagrantfile",
        "//:version",
        "//cluster:all-srcs",
        "//docs:all-srcs",
        "//examples:all-srcs",
        "//hack/lib:all-srcs",
        "//third_party/htpasswd:all-srcs",
    ],
    package_dir = "kubernetes",
    strip_prefix = "//",
    deps = [
        ":_full_server",
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

release_filegroup(
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
