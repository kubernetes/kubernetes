package(default_visibility = ["//visibility:public"])

load("@io_k8s_repo_infra//defs:pkg.bzl", "pkg_tar")

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
    srcs = [
        ":addon-srcs",
    ],
    extension = "tar.gz",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
