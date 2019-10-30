load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "conversion_generator.go",
        "generic_conversion.go",
        "manual_conversions_tracker.go",
        "unsafe_conversion_arbitrator.go",
        "utils.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/code-generator/cmd/generic-conversion-gen/generators",
    importpath = "k8s.io/code-generator/cmd/generic-conversion-gen/generators",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/code-generator/cmd/generic-conversion-gen/args:go_default_library",
        "//vendor/k8s.io/gengo/args:go_default_library",
        "//vendor/k8s.io/gengo/generator:go_default_library",
        "//vendor/k8s.io/gengo/namer:go_default_library",
        "//vendor/k8s.io/gengo/types:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
