load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["deep_equal_test.go"],
    embed = [":go_default_library"],
)

go_library(
    name = "go_default_library",
    srcs = ["deep_equal.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apimachinery/third_party/forked/golang/reflect",
    importpath = "k8s.io/apimachinery/third_party/forked/golang/reflect",
    visibility = ["//staging/src/k8s.io/apimachinery/pkg/conversion:__pkg__"],
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
