package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "generic_soak.go",
        "imports.go",
    ],
    importpath = "k8s.io/kubernetes/test/e2e/instrumentation/logging",
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//test/e2e/framework/config:go_default_library",
        "//test/e2e/instrumentation/common:go_default_library",
        "//test/e2e/instrumentation/logging/elasticsearch:go_default_library",
        "//test/e2e/instrumentation/logging/stackdriver:go_default_library",
        "//test/utils/image:go_default_library",
        "//vendor/github.com/onsi/ginkgo:go_default_library",
        "//vendor/github.com/onsi/gomega:go_default_library",
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
        "//test/e2e/instrumentation/logging/elasticsearch:all-srcs",
        "//test/e2e/instrumentation/logging/stackdriver:all-srcs",
        "//test/e2e/instrumentation/logging/utils:all-srcs",
    ],
    tags = ["automanaged"],
)
