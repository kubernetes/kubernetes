load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "converter.go",
        "discovery.go",
        "interfaces.go",
        "multi_client.go",
        "versioned_client.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/metrics/pkg/client/custom_metrics",
    importpath = "k8s.io/metrics/pkg/client/custom_metrics",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/client-go/discovery:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/util/flowcontrol:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/custom_metrics:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/custom_metrics/v1beta1:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/custom_metrics/v1beta2:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/client/custom_metrics/scheme:go_default_library",
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
        "//staging/src/k8s.io/metrics/pkg/client/custom_metrics/fake:all-srcs",
        "//staging/src/k8s.io/metrics/pkg/client/custom_metrics/scheme:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["util_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/custom_metrics:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/custom_metrics/v1beta1:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/custom_metrics/v1beta2:go_default_library",
        "//vendor/github.com/stretchr/testify/require:go_default_library",
    ],
)
