load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["resource_usage_gatherer.go"],
    importpath = "k8s.io/kubernetes/test/e2e/scheduling/gatherer",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//test/e2e/framework:go_default_library",
        "//test/e2e/framework/kubelet:go_default_library",
        "//test/e2e/system:go_default_library",
    ],
)
