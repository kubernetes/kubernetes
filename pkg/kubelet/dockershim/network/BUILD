load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "network.go",
        "plugins.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/dockershim/network",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/features:go_default_library",
        "//pkg/kubelet/apis/config:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//pkg/kubelet/dockershim/network/hostport:go_default_library",
        "//pkg/kubelet/dockershim/network/metrics:go_default_library",
        "//pkg/util/sysctl:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
        "//vendor/k8s.io/utils/exec:go_default_library",
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
        "//pkg/kubelet/dockershim/network/cni:all-srcs",
        "//pkg/kubelet/dockershim/network/hairpin:all-srcs",
        "//pkg/kubelet/dockershim/network/hostport:all-srcs",
        "//pkg/kubelet/dockershim/network/kubenet:all-srcs",
        "//pkg/kubelet/dockershim/network/metrics:all-srcs",
        "//pkg/kubelet/dockershim/network/testing:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["plugins_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/kubelet/dockershim/network/metrics:go_default_library",
        "//staging/src/k8s.io/component-base/metrics/legacyregistry:go_default_library",
        "//staging/src/k8s.io/component-base/metrics/testutil:go_default_library",
    ],
)
