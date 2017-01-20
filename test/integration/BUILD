package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "utils.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/core/v1:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
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
        "//test/integration/discoverysummarizer:all-srcs",
        "//test/integration/examples:all-srcs",
        "//test/integration/federation:all-srcs",
        "//test/integration/framework:all-srcs",
        "//test/integration/metrics:all-srcs",
        "//test/integration/objectmeta:all-srcs",
        "//test/integration/openshift:all-srcs",
        "//test/integration/scheduler_perf:all-srcs",
    ],
    tags = ["automanaged"],
)
