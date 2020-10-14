load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "error_channel.go",
        "parallelism.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/scheduler/internal/parallelize",
    visibility = ["//pkg/scheduler:__subpackages__"],
    deps = ["//staging/src/k8s.io/client-go/util/workqueue:go_default_library"],
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
    srcs = [
        "error_channel_test.go",
        "parallelism_test.go",
    ],
    embed = [":go_default_library"],
)
