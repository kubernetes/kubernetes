load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "actual_state_of_world.go",
        "desired_state_of_world.go",
        "types.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/pluginmanager/cache",
    visibility = ["//visibility:public"],
    deps = ["//vendor/k8s.io/klog:go_default_library"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "actual_state_of_world_test.go",
        "desired_state_of_world_test.go",
    ],
    embed = [":go_default_library"],
    deps = ["//vendor/github.com/stretchr/testify/require:go_default_library"],
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
