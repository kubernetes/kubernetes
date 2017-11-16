package(default_visibility = ["//visibility:public"])

load("@io_bazel//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

filegroup(
    name = "addon-srcs",
    srcs = glob(
        [
            "**/*.json",
            "**/*.yaml",
            "**/*.yaml.in",
        ],
        exclude = ["**/*demo*/**"],
    ),
)

pkg_tar(
    name = "addons",
    extension = "tar.gz",
    files = [
        ":addon-srcs",
    ],
    mode = "0644",
    strip_prefix = ".",
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
        "//cluster/addons/fluentd-elasticsearch/es-image:all-srcs",
    ],
    tags = ["automanaged"],
)
