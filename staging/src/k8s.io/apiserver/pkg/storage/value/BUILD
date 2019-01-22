package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["transformer_test.go"],
    embed = [":go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = [
        "metrics.go",
        "transformer.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/storage/value",
    importpath = "k8s.io/apiserver/pkg/storage/value",
    deps = ["//vendor/github.com/prometheus/client_golang/prometheus:go_default_library"],
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
        "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/aes:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/envelope:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/identity:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/storage/value/encrypt/secretbox:all-srcs",
    ],
    tags = ["automanaged"],
)
