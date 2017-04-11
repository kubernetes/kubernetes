package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

licenses(["notice"])

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//cluster/addons:all-srcs",
        "//cluster/gce:all-srcs",
        "//cluster/saltbase:all-srcs",
    ],
    tags = ["automanaged"],
)

# All of the manifests that are expected to be in a "gci-trusty"
# subdir of the manifests tarball.
pkg_tar(
    name = "_manifests-gci-trusty",
    package_dir = "gci-trusty",
    visibility = ["//visibility:private"],
    deps = [
        "//cluster/addons",
        "//cluster/gce:gci-trusty-manifests",
        "//cluster/saltbase:gci-trusty-salt-manifests",
    ],
)

pkg_tar(
    name = "manifests",
    mode = "0644",
    package_dir = "kubernetes",
    deps = [
        ":_manifests-gci-trusty",
        "//cluster/saltbase:salt-manifests",
    ],
)
