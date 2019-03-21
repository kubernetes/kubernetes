package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "balanced_resource_allocation.go",
        "image_locality.go",
        "interpod_affinity.go",
        "least_requested.go",
        "metadata.go",
        "most_requested.go",
        "node_affinity.go",
        "node_label.go",
        "node_prefer_avoid_pods.go",
        "priorities.go",
        "reduce.go",
        "requested_to_capacity_ratio.go",
        "resource_allocation.go",
        "resource_limits.go",
        "selector_spreading.go",
        "taint_toleration.go",
        "test_util.go",
        "types.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/scheduler/algorithm/priorities",
    deps = [
        "//pkg/apis/core/v1/helper:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/scheduler/algorithm:go_default_library",
        "//pkg/scheduler/algorithm/predicates:go_default_library",
        "//pkg/scheduler/algorithm/priorities/util:go_default_library",
        "//pkg/scheduler/api:go_default_library",
        "//pkg/scheduler/nodeinfo:go_default_library",
        "//pkg/util/node:go_default_library",
        "//pkg/util/parsers:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/client-go/util/workqueue:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "balanced_resource_allocation_test.go",
        "image_locality_test.go",
        "interpod_affinity_test.go",
        "least_requested_test.go",
        "metadata_test.go",
        "most_requested_test.go",
        "node_affinity_test.go",
        "node_label_test.go",
        "node_prefer_avoid_pods_test.go",
        "requested_to_capacity_ratio_test.go",
        "resource_limits_test.go",
        "selector_spreading_test.go",
        "taint_toleration_test.go",
        "types_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/features:go_default_library",
        "//pkg/scheduler/algorithm:go_default_library",
        "//pkg/scheduler/algorithm/priorities/util:go_default_library",
        "//pkg/scheduler/api:go_default_library",
        "//pkg/scheduler/nodeinfo:go_default_library",
        "//pkg/scheduler/testing:go_default_library",
        "//pkg/util/parsers:go_default_library",
        "//staging/src/k8s.io/api/apps/v1:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature/testing:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
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
        "//pkg/scheduler/algorithm/priorities/util:all-srcs",
    ],
    tags = ["automanaged"],
)
