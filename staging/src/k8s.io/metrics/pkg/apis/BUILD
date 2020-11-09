load("@io_bazel_rules_go//go:def.bzl", "go_test")

go_test(
    name = "go_default_test",
    srcs = ["roundtrip_test.go"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/apitesting/fuzzer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/apitesting/roundtrip:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/fuzzer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/custom_metrics:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/custom_metrics/v1beta1:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/custom_metrics/v1beta2:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/external_metrics:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/external_metrics/v1beta1:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/metrics:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/metrics/v1alpha1:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/apis/metrics/v1beta1:go_default_library",
        "//vendor/github.com/stretchr/testify/require:go_default_library",
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
        "//staging/src/k8s.io/metrics/pkg/apis/custom_metrics:all-srcs",
        "//staging/src/k8s.io/metrics/pkg/apis/external_metrics:all-srcs",
        "//staging/src/k8s.io/metrics/pkg/apis/metrics:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
