load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["kv.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/cli-runtime/pkg/kustomize/k8sdeps/kv",
    importpath = "k8s.io/cli-runtime/pkg/kustomize/k8sdeps/kv",
    visibility = ["//visibility:public"],
    deps = ["//staging/src/k8s.io/apimachinery/pkg/util/validation:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = ["kv_test.go"],
    embed = [":go_default_library"],
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
