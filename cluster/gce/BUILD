package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

pkg_tar(
    name = "gci-trusty-manifests",
    files = [
        "container-linux/configure-helper.sh",
        "gci/configure-helper.sh",
        "gci/health-monitor.sh",
        "gci/mounter/mounter",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
