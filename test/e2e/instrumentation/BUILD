package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["imports.go"],
    importpath = "k8s.io/kubernetes/test/e2e/instrumentation",
    deps = [
        "//test/e2e/instrumentation/logging:go_default_library",
        "//test/e2e/instrumentation/monitoring:go_default_library",
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
        "//test/e2e/instrumentation/common:all-srcs",
        "//test/e2e/instrumentation/logging:all-srcs",
        "//test/e2e/instrumentation/monitoring:all-srcs",
    ],
    tags = ["automanaged"],
)
