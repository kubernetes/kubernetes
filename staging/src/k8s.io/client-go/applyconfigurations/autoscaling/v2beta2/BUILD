load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "containerresourcemetricsource.go",
        "containerresourcemetricstatus.go",
        "crossversionobjectreference.go",
        "externalmetricsource.go",
        "externalmetricstatus.go",
        "horizontalpodautoscaler.go",
        "horizontalpodautoscalerbehavior.go",
        "horizontalpodautoscalercondition.go",
        "horizontalpodautoscalerspec.go",
        "horizontalpodautoscalerstatus.go",
        "hpascalingpolicy.go",
        "hpascalingrules.go",
        "metricidentifier.go",
        "metricspec.go",
        "metricstatus.go",
        "metrictarget.go",
        "metricvaluestatus.go",
        "objectmetricsource.go",
        "objectmetricstatus.go",
        "podsmetricsource.go",
        "podsmetricstatus.go",
        "resourcemetricsource.go",
        "resourcemetricstatus.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/client-go/applyconfigurations/autoscaling/v2beta2",
    importpath = "k8s.io/client-go/applyconfigurations/autoscaling/v2beta2",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/autoscaling/v2beta2:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/client-go/applyconfigurations/meta/v1:go_default_library",
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
