load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["crd.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/csi-api/pkg/crd",
    importpath = "k8s.io/csi-api/pkg/crd",
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["crd_test.go"],
    data = [
        ":csi-manifests",
        "//cluster/addons:addon-srcs",
    ],
    embed = [":go_default_library"],
    deps = ["//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library"],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "csi-manifests",
    srcs = glob(["manifests/**"]),
    visibility = ["//visibility:public"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
