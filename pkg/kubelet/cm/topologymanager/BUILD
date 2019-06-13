load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "policy.go",
        "policy_preferred.go",
        "policy_strict.go",
        "topology_manager.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/cm/topologymanager",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubelet/cm/topologymanager/socketmask:go_default_library",
        "//pkg/kubelet/lifecycle:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
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
        "//pkg/kubelet/cm/topologymanager/socketmask:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "policy_preferred_test.go",
        "policy_strict_test.go",
    ],
    embed = [":go_default_library"],
)
