package(default_visibility = ["//visibility:public"])

load("@bazel_tools//tools/build_defs/pkg:pkg.bzl", "pkg_tar")

filegroup(
    name = "addon-srcs",
    srcs = glob([
        "calico-policy-controller/*",
        "cluster-loadbalancing/*",
        "cluster-monitoring/*",
        "dashboard/*",
        "dns/*",
        "etcd-empty-dir-cleanup/*",
        "fluentd-elasticsearch/*",
        "fluentd-gcp/*",
        "gci/*",
        "node-problem-detector/*",
        "podsecuritypolicies/*",
        "python-image/*",
        "registry/*",
    ]),
)

pkg_tar(
    name = "addons",
    extension = "tar.gz",
    files = [
        ":addon-srcs",
    ],
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
