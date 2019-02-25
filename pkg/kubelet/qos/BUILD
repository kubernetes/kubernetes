package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["policy_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/apis/scheduling:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "policy.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/qos",
    deps = [
        "//pkg/apis/core/v1/helper/qos:go_default_library",
        "//pkg/kubelet/types:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
