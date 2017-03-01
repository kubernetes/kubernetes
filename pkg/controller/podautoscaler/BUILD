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
        "doc.go",
        "horizontal.go",
        "replica_calculator.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/autoscaling/v1:go_default_library",
        "//pkg/apis/autoscaling/v2alpha1:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/autoscaling/v1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/core/v1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/extensions/v1beta1:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/autoscaling/v1:go_default_library",
        "//pkg/client/listers/autoscaling/v1:go_default_library",
        "//pkg/controller/podautoscaler/metrics:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/client-go/kubernetes/typed/core/v1",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/tools/cache",
        "//vendor:k8s.io/client-go/tools/record",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "horizontal_test.go",
        "legacy_horizontal_test.go",
        "legacy_replica_calculator_test.go",
        "replica_calculator_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/unversioned:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/autoscaling/install:go_default_library",
        "//pkg/apis/autoscaling/v1:go_default_library",
        "//pkg/apis/autoscaling/v2alpha1:go_default_library",
        "//pkg/apis/extensions/install:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset/fake:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions:go_default_library",
        "//pkg/controller:go_default_library",
        "//pkg/controller/podautoscaler/metrics:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:github.com/stretchr/testify/require",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/kubernetes/fake",
        "//vendor:k8s.io/client-go/pkg/api",
        "//vendor:k8s.io/client-go/pkg/api/v1",
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
    srcs = [
        ":package-srcs",
        "//pkg/controller/podautoscaler/metrics:all-srcs",
    ],
    tags = ["automanaged"],
)
