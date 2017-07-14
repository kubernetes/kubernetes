package(default_visibility = ["//visibility:public"])

load("@io_bazel//tools/build_defs/pkg:pkg.bzl", "pkg_tar")
load("@io_kubernetes_build//defs:build.bzl", "release_filegroup")

pkg_tar(
    name = "gci-trusty-manifests",
    files = [
        "container-linux/configure-helper.sh",
        "gci/configure-helper.sh",
        "gci/health-monitor.sh",
        "//cluster/gce/gci/mounter",
    ],
    mode = "0755",
    strip_prefix = ".",
    # pkg_tar doesn't support renaming the files we add, so instead create symlinks.
    symlinks = {
        "container-linux-configure-helper.sh": "container-linux/configure-helper.sh",
        "gci-configure-helper.sh": "gci/configure-helper.sh",
        "health-monitor.sh": "gci/health-monitor.sh",
        "gci-mounter": "gci/mounter/mounter",
        "trusty-configure-helper.sh": "trusty/configure-helper.sh",
    },
)

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
        "//cluster/gce/gci/mounter:all-srcs",
    ],
    tags = ["automanaged"],
)

# Having the configure-vm.sh script and and trusty code from the GCE cluster
# deploy hosted with the release is useful for GKE.
# This list should match the list in kubernetes/release/lib/releaselib.sh.
release_filegroup(
    name = "gcs-release-artifacts",
    srcs = [
        "configure-vm.sh",
        "gci/configure.sh",
        "gci/master.yaml",
        "gci/node.yaml",
    ],
)
