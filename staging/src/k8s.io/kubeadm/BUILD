package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
)
load("//pkg/version:def.bzl", "version_x_defs")

go_binary(
    name = "kubeadm",
    gc_linkopts = [
        "-linkmode",
        "external",
        "-extldflags",
        "-static",
    ],
    library = ":go_default_library",
    x_defs = version_x_defs(),
)

go_library(
    name = "go_default_library",
    srcs = ["kubeadm.go"],
    deps = ["//vendor/k8s.io/kubeadm/app:go_default_library"],
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
        "//staging/src/k8s.io/kubeadm/app:all-srcs",
        "//staging/src/k8s.io/kubeadm/test:all-srcs",
    ],
    tags = ["automanaged"],
)
