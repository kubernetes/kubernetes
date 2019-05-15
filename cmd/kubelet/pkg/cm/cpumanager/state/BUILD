load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "checkpoint.go",
        "state.go",
        "state_checkpoint.go",
        "state_file.go",
        "state_mem.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/kubelet/pkg/cm/cpumanager/state",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/kubelet/pkg/checkpointmanager:go_default_library",
        "//cmd/kubelet/pkg/checkpointmanager/checksum:go_default_library",
        "//cmd/kubelet/pkg/checkpointmanager/errors:go_default_library",
        "//cmd/kubelet/pkg/cm/cpuset:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "state_checkpoint_test.go",
        "state_compatibility_test.go",
        "state_file_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//cmd/kubelet/pkg/checkpointmanager:go_default_library",
        "//cmd/kubelet/pkg/cm/cpumanager/state/testing:go_default_library",
        "//cmd/kubelet/pkg/cm/cpuset:go_default_library",
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
        "//cmd/kubelet/pkg/cm/cpumanager/state/testing:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
