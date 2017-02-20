package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "fake_cache.go",
        "pods_to_cache.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//plugin/pkg/scheduler/schedulercache:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apis/pkg/api/v1",
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
