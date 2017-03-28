package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "interfaces.go",
        "legacy_metrics_client.go",
        "rest_metrics_client.go",
        "utilization.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/autoscaling/v2alpha1:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/core/v1:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/heapster/metrics/api/v1/types",
        "//vendor:k8s.io/heapster/metrics/apis/metrics/v1alpha1",
        "//vendor:k8s.io/metrics/pkg/apis/custom_metrics/v1alpha1",
        "//vendor:k8s.io/metrics/pkg/client/clientset_generated/clientset/typed/metrics/v1alpha1",
        "//vendor:k8s.io/metrics/pkg/client/custom_metrics",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "legacy_metrics_client_test.go",
        "rest_metrics_client_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/unversioned:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/autoscaling/v2alpha1:go_default_library",
        "//pkg/client/clientset_generated/clientset/fake:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/install",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/pkg/apis/extensions/install",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/testing",
        "//vendor:k8s.io/heapster/metrics/api/v1/types",
        "//vendor:k8s.io/heapster/metrics/apis/metrics/v1alpha1",
        "//vendor:k8s.io/metrics/pkg/apis/custom_metrics/v1alpha1",
        "//vendor:k8s.io/metrics/pkg/apis/metrics/v1alpha1",
        "//vendor:k8s.io/metrics/pkg/client/clientset_generated/clientset/fake",
        "//vendor:k8s.io/metrics/pkg/client/custom_metrics/fake",
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
