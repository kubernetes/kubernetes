package(default_visibility = ["//visibility:public"])

load("@io_k8s_repo_infra//defs:build.bzl", "release_filegroup")
load("@io_k8s_repo_infra//defs:pkg.bzl", "pkg_tar")

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
        "//cluster/gce/windows:all-srcs",
    ],
    tags = ["automanaged"],
)
