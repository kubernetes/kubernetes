package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = ["features.go"],
    importpath = "k8s.io/kubernetes/cmd/kubeadm/app/features",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/version:go_default_library",
        "//staging/src/k8s.io/component-base/featuregate:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = ["features_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//cmd/kubeadm/app/constants:go_default_library",
        "//staging/src/k8s.io/component-base/featuregate:go_default_library",
    ],
)
