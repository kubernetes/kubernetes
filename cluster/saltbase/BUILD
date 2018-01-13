package(default_visibility = ["//visibility:public"])

load("@io_bazel//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

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

# TODO(#3579): This is a temporary hack. It gathers up the yaml,
# yaml.in, json files in cluster/addons (minus any demos) and overlays
# them into kube-addons, where we expect them.
# These files are expected in a salt/kube-addons subdirectory.
pkg_tar(
    name = "_salt_kube-addons",
    package_dir = "salt/kube-addons",
    strip_prefix = "/cluster/addons",
    visibility = ["//visibility:private"],
    deps = [
        "//cluster/addons",
    ],
)

pkg_tar(
    name = "salt",
    files = glob(
        ["**"],
        exclude = ["BUILD"],
    ),
    mode = "0644",
    modes = {
        "install.sh": "0755",
    },
    package_dir = "kubernetes/saltbase",
    strip_prefix = ".",
    deps = [
        ":_salt_kube-addons",
    ],
)
