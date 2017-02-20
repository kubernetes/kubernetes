package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "cache.go",
        "interface.go",
        "node_info.go",
        "reconcile_affinity.go",
        "util.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/features:go_default_library",
        "//plugin/pkg/scheduler/algorithm/priorities/util:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "cache_test.go",
        "reconcile_affinity_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//plugin/pkg/scheduler/algorithm/priorities/util:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
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
)
