package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["openshift_test.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/genericapiserver:go_default_library",
        "//pkg/master:go_default_library",
    ],
)
