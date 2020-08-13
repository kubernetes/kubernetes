load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["no-restraint.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/testing",
    importpath = "k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/testing",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/debug:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/metrics:go_default_library",
    ],
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
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/testing/clock:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
