package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["plugins.go"],
    importpath = "k8s.io/kubernetes/pkg/scheduler/algorithmprovider",
    deps = ["//pkg/scheduler/algorithmprovider/defaults:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = ["plugins_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/features:go_default_library",
        "//pkg/scheduler/factory:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/component-base/featuregate/testing:go_default_library",
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
        "//pkg/scheduler/algorithmprovider/defaults:all-srcs",
    ],
    tags = ["automanaged"],
)
