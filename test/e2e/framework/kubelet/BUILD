load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "kubelet_pods.go",
        "stats.go",
    ],
    importpath = "k8s.io/kubernetes/test/e2e/framework/kubelet",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubelet/apis/stats/v1alpha1:go_default_library",
        "//pkg/kubelet/dockershim/metrics:go_default_library",
        "//pkg/master/ports:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//test/e2e/framework/log:go_default_library",
        "//test/e2e/framework/metrics:go_default_library",
        "//test/e2e/framework/node:go_default_library",
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
    visibility = ["//visibility:public"],
)
