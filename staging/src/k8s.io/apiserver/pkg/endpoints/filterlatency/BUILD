load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["filterlatency.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/endpoints/filterlatency",
    importpath = "k8s.io/apiserver/pkg/endpoints/filterlatency",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/clock:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/endpoints/metrics:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/endpoints/request:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = ["filterlatency_test.go"],
    embed = [":go_default_library"],
    deps = ["//staging/src/k8s.io/apimachinery/pkg/util/clock:go_default_library"],
)
