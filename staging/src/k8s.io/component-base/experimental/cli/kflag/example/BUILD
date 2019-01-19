load("@io_bazel_rules_go//go:def.bzl", "go_binary", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["example.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/component-base/experimental/cli/kflag/example",
    importpath = "k8s.io/component-base/experimental/cli/kflag/example",
    visibility = ["//visibility:private"],
    deps = [
        "//staging/src/k8s.io/component-base/experimental/cli/kflag:go_default_library",
        "//staging/src/k8s.io/component-base/experimental/cli/kflag/example/config:go_default_library",
        "//staging/src/k8s.io/component-base/experimental/cli/kflag/example/options:go_default_library",
    ],
)

go_binary(
    name = "example",
    embed = [":go_default_library"],
    visibility = ["//visibility:public"],
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
        "//staging/src/k8s.io/component-base/experimental/cli/kflag/example/config:all-srcs",
        "//staging/src/k8s.io/component-base/experimental/cli/kflag/example/options:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
