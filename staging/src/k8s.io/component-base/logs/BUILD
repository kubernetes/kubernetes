package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "logs.go",
        "options.go",
        "registry.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/component-base/logs",
    importpath = "k8s.io/component-base/logs",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/component-base/logs/json:go_default_library",
        "//vendor/github.com/go-logr/logr:go_default_library",
        "//vendor/github.com/spf13/pflag:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
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
        "//staging/src/k8s.io/component-base/logs/json:all-srcs",
        "//staging/src/k8s.io/component-base/logs/logreduction:all-srcs",
    ],
    tags = ["automanaged"],
)
