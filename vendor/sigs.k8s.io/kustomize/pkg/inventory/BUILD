load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "constants.go",
        "inventory.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/pkg/inventory",
    importpath = "sigs.k8s.io/kustomize/pkg/inventory",
    visibility = ["//visibility:public"],
    deps = ["//vendor/sigs.k8s.io/kustomize/pkg/resid:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = ["inventory_test.go"],
    embed = [":go_default_library"],
    tags = ["manual"],
    deps = ["//vendor/sigs.k8s.io/kustomize/pkg/resid:go_default_library"],
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
