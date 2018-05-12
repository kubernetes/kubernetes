package(default_visibility = ["//visibility:public"])

load("@io_kubernetes_build//defs:build.bzl", "release_filegroup")
load("@io_kubernetes_build//defs:pkg.bzl", "pkg_tar")

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
        "//cluster/gce/addons:all-srcs",
        "//cluster/gce/gci:all-srcs",
        "//cluster/gce/manifests:all-srcs",
    ],
    tags = ["automanaged"],
)
