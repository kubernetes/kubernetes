package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["doc.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/controller-manager",
    importpath = "k8s.io/controller-manager",
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
        "//staging/src/k8s.io/controller-manager/app:all-srcs",
        "//staging/src/k8s.io/controller-manager/config:all-srcs",
        "//staging/src/k8s.io/controller-manager/options:all-srcs",
        "//staging/src/k8s.io/controller-manager/pkg/clientbuilder:all-srcs",
        "//staging/src/k8s.io/controller-manager/pkg/features:all-srcs",
        "//staging/src/k8s.io/controller-manager/pkg/informerfactory:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
