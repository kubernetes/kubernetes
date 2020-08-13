package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "cadvisor_e2e.go",
        "node_conformance.go",
        "node_e2e.go",
        "remote.go",
        "ssh.go",
        "types.go",
        "utils.go",
    ],
    importpath = "k8s.io/kubernetes/test/e2e_node/remote",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//test/e2e_node/builder:go_default_library",
        "//test/e2e_node/system:go_default_library",
        "//test/utils:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
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
)
