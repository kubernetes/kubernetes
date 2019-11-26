load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["interface.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing",
    importpath = "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing",
    visibility = ["//visibility:public"],
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
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/promise:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/queueset:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/testing:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
