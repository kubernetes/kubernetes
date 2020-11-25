load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "interfaces.go",
        "legacy_metrics_client.go",
        "rest_metrics_client.go",
        "utilization.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/controller/podautoscaler/metrics",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/autoscaling/v2beta2:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/typed/core/v1:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/custom_metrics/v1beta2:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/metrics/v1alpha1:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/metrics/v1beta1:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/client/clientset/versioned/typed/metrics/v1beta1:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/client/custom_metrics:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/client/external_metrics:go_default_library",
        "//vendor/k8s.io/heapster/metrics/api/v1/types:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "legacy_metrics_client_test.go",
        "rest_metrics_client_test.go",
        "utilization_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/api/legacyscheme:go_default_library",
        "//pkg/apis/apps/install:go_default_library",
        "//staging/src/k8s.io/api/autoscaling/v2beta2:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/meta/testrestmapper:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/fake:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/testing:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/custom_metrics/v1beta2:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/external_metrics/v1beta1:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/metrics/v1alpha1:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/metrics/v1beta1:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/client/clientset/versioned/fake:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/client/custom_metrics/fake:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/client/external_metrics/fake:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/k8s.io/heapster/metrics/api/v1/types:go_default_library",
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
