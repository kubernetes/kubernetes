load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["types.go"],
    importpath = "k8s.io/kubernetes/test/conformance/behaviors",
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["behaviors_test.go"],
    data = glob(["*/*.yaml"]),
    embed = [":go_default_library"],
    deps = [
        "//vendor/gopkg.in/yaml.v2:go_default_library",
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
