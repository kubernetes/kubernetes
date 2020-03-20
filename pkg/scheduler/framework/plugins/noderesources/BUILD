load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "balanced_allocation.go",
        "fit.go",
        "least_allocated.go",
        "most_allocated.go",
        "node_resource_topology_match.go",
        "requested_to_capacity_ratio.go",
        "resource_allocation.go",
        "test_util.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/scheduler/framework/plugins/noderesources",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/apis/core/v1/helper:go_default_library",
        "//pkg/apis/core/v1/helper/qos:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/kubelet/cm/topologymanager/bitmask:go_default_library",
        "//pkg/scheduler/apis/config:go_default_library",
        "//pkg/scheduler/apis/config/validation:go_default_library",
        "//pkg/scheduler/framework:go_default_library",
        "//pkg/scheduler/util:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1/validation:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/client-go/tools/clientcmd:go_default_library",
        "//staging/src/k8s.io/client-go/tools/clientcmd/api:go_default_library",
        "//staging/src/k8s.io/noderesourcetopology-api/pkg/apis/topology/v1alpha1:go_default_library",
        "//staging/src/k8s.io/noderesourcetopology-api/pkg/generated/clientset/versioned:go_default_library",
        "//staging/src/k8s.io/noderesourcetopology-api/pkg/generated/informers/externalversions:go_default_library",
        "//staging/src/k8s.io/noderesourcetopology-api/pkg/generated/informers/externalversions/topology/v1alpha1:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
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

go_test(
    name = "go_default_test",
    srcs = [
        "balanced_allocation_test.go",
        "fit_test.go",
        "least_allocated_test.go",
        "most_allocated_test.go",
        "requested_to_capacity_ratio_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/apis/core/v1/helper:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/scheduler/apis/config:go_default_library",
        "//pkg/scheduler/framework:go_default_library",
        "//pkg/scheduler/framework/runtime:go_default_library",
        "//pkg/scheduler/internal/cache:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/intstr:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/component-base/featuregate:go_default_library",
        "//staging/src/k8s.io/component-base/featuregate/testing:go_default_library",
        "//staging/src/k8s.io/noderesourcetopology-api/pkg/apis/topology/v1alpha1:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
    ],
)
