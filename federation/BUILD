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
    srcs = [
        ":package-srcs",
        "//federation/apis/core:all-srcs",
        "//federation/apis/federation:all-srcs",
        "//federation/client/cache:all-srcs",
        "//federation/client/clientset_generated/federation_clientset:all-srcs",
        "//federation/cluster:all-srcs",
        "//federation/cmd/federation-apiserver:all-srcs",
        "//federation/cmd/federation-controller-manager:all-srcs",
        "//federation/cmd/genfeddocs:all-srcs",
        "//federation/cmd/kubefed:all-srcs",
        "//federation/develop:all-srcs",
        "//federation/pkg/dnsprovider:all-srcs",
        "//federation/pkg/federatedtypes:all-srcs",
        "//federation/pkg/federation-controller:all-srcs",
        "//federation/pkg/kubefed:all-srcs",
        "//federation/plugin/pkg/admission/schedulingpolicy:all-srcs",
        "//federation/registry/cluster:all-srcs",
    ],
    tags = ["automanaged"],
)

pkg_tar(
    name = "release",
    files = glob([
        "deploy/**",
        "manifests/**",
    ]) + ["//federation/cluster:all-srcs"],
    package_dir = "federation",
)
