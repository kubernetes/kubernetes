package(default_visibility = ["//visibility:public"])

load(
    "//build:platforms.bzl",
    "CLIENT_PLATFORMS",
    "NODE_PLATFORMS",
    "SERVER_PLATFORMS",
    "TEST_PLATFORMS",
    "for_platforms",
    "go_platform_constraint",
)
load("@io_k8s_repo_infra//defs:build.bzl", "release_filegroup")
load("@io_k8s_repo_infra//defs:pkg.bzl", "pkg_tar")
load("//staging:repos_generated.bzl", "staging_repos")

# Bazel doesn't make the output filename
# (such as kubernetes-server-{OS}-{ARCH}.tar.gz) configurable, so we instead
# create rules for all platforms and tag them manual.
# We then select the correct set of platform-specific tarballs in this filegroup
# using a select() statement.
# Thus the release-tars target always selects the correct set of tarballs
# for the configured platform being built.
release_filegroup(
    name = "release-tars",
    conditioned_srcs = for_platforms(
        for_all = [
            ":kubernetes.tar.gz",
            ":kubernetes-src.tar.gz",
        ],
        for_client = [":kubernetes-client-{OS}-{ARCH}.tar.gz"],
        for_node = [":kubernetes-node-{OS}-{ARCH}.tar.gz"],
        for_server = [
            ":kubernetes-server-{OS}-{ARCH}.tar.gz",
            ":kubernetes-manifests.tar.gz",
        ],
        for_test = [
            ":kubernetes-test-portable.tar.gz",
            ":kubernetes-test-{OS}-{ARCH}.tar.gz",
        ],
    ),
)

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

pkg_tar(
    name = "kubernetes-src",
    srcs = ["//:all-srcs"],
    extension = "tar.gz",
    package_dir = "kubernetes",
    strip_prefix = "//",
    symlinks = {
        "kubernetes/vendor/%s" % repo: "../../staging/src/%s" % repo
        for repo in staging_repos
    },
    tags = [
        "manual",
        "no-cache",
    ],
)

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
    srcs = ["//build:client-targets"],
    mode = "0755",
    package_dir = "client/bin",
    tags = [
        "manual",
        "no-cache",
    ],
    visibility = ["//visibility:private"],
)

[[pkg_tar(
    name = "kubernetes-client-%s-%s" % (os, arch),
    extension = "tar.gz",
    package_dir = "kubernetes",
    tags = [
        "manual",
        "no-cache",
    ],
    deps = select({go_platform_constraint(os, arch): [":_client-bin"]}),
) for arch in archs] for os, archs in CLIENT_PLATFORMS.items()]

pkg_tar(
    name = "_node-bin",
    srcs = [
        "//build:client-targets",
        "//build:node-targets",
    ],
    mode = "0755",
    package_dir = "node/bin",
    tags = [
        "manual",
        "no-cache",
    ],
    visibility = ["//visibility:private"],
)

[[pkg_tar(
    name = "kubernetes-node-%s-%s" % (os, arch),
    srcs = [":license-targets"],
    extension = "tar.gz",
    mode = "0644",
    package_dir = "kubernetes",
    tags = [
        "manual",
        "no-cache",
    ],
    deps = select({go_platform_constraint(os, arch): [":_node-bin"]}),
) for arch in archs] for os, archs in NODE_PLATFORMS.items()]

pkg_tar(
    name = "_server-bin",
    srcs = [
        "//build:client-targets",
        "//build:docker-artifacts",
        "//build:node-targets",
        "//build:server-targets",
    ],
    mode = "0755",
    package_dir = "server/bin",
    tags = [
        "manual",
        "no-cache",
    ],
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
    srcs = [
        ":.dummy",
    ],
    package_dir = "addons",
    tags = [
        "manual",
        "no-cache",
    ],
    visibility = ["//visibility:private"],
)

[[pkg_tar(
    name = "kubernetes-server-%s-%s" % (os, arch),
    srcs = [":license-targets"],
    extension = "tar.gz",
    mode = "0644",
    package_dir = "kubernetes",
    tags = [
        "manual",
        "no-cache",
    ],
    deps = select({
        go_platform_constraint(os, arch): [
            ":_server-addons",
            ":_server-bin",
        ],
    }),
) for arch in archs] for os, archs in SERVER_PLATFORMS.items()]

pkg_tar(
    name = "kubernetes-test-portable",
    srcs = ["//build:test-portable-targets"],
    extension = "tar.gz",
    package_dir = "kubernetes",
    strip_prefix = "//",
    tags = ["no-cache"],
)

pkg_tar(
    name = "_test-bin",
    srcs = ["//build:test-targets"],
    mode = "0755",
    package_dir = "test/bin",
    tags = [
        "manual",
        "no-cache",
    ],
    visibility = ["//visibility:private"],
)

[[pkg_tar(
    name = "kubernetes-test-%s-%s" % (os, arch),
    extension = "tar.gz",
    package_dir = "kubernetes",
    strip_prefix = "//",
    tags = [
        "manual",
        "no-cache",
    ],
    deps = select({go_platform_constraint(os, arch): [":_test-bin"]}),
) for arch in archs] for os, archs in TEST_PLATFORMS.items()]

pkg_tar(
    name = "_full_server",
    srcs = [
        ":kubernetes-manifests.tar.gz",
    ],
    package_dir = "server",
    tags = [
        "manual",
        "no-cache",
    ],
    visibility = ["//visibility:private"],
)

pkg_tar(
    name = "kubernetes",
    srcs = [
        "//:Godeps/LICENSES",
        "//:README.md",
        "//:version",
        "//cluster:all-srcs",
        "//docs:all-srcs",
        "//hack/lib:all-srcs",
    ],
    extension = "tar.gz",
    package_dir = "kubernetes",
    strip_prefix = "//",
    symlinks = {
        "kubernetes/cluster/gce/cos/": "gci",
        "kubernetes/cluster/gce/custom/": "gci",
        "kubernetes/cluster/gce/ubuntu/": "gci",
    },
    tags = ["no-cache"],
    deps = [
        ":_full_server",
    ],
)

pkg_tar(
    name = "kubernetes-manifests",
    extension = "tar.gz",
    tags = ["no-cache"],
    deps = [
        "//cluster:manifests",
    ],
)
