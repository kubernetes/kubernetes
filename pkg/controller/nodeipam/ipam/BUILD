package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "cloud_cidr_allocator_test.go",
        "controller_test.go",
        "range_allocator_test.go",
        "timeout_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/controller:go_default_library",
        "//pkg/controller/nodeipam/ipam/cidrset:go_default_library",
        "//pkg/controller/nodeipam/ipam/test:go_default_library",
        "//pkg/controller/testutil:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/informers/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/fake:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "adapter.go",
        "cidr_allocator.go",
        "cloud_cidr_allocator.go",
        "controller.go",
        "doc.go",
        "range_allocator.go",
        "timeout.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/controller/nodeipam/ipam",
    deps = [
        "//pkg/controller:go_default_library",
        "//pkg/controller/nodeipam/ipam/cidrset:go_default_library",
        "//pkg/controller/nodeipam/ipam/sync:go_default_library",
        "//pkg/controller/util/node:go_default_library",
        "//pkg/scheduler/api:go_default_library",
        "//pkg/util/node:go_default_library",
        "//pkg/util/taints:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/client-go/informers/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/scheme:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/typed/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/listers/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        "//staging/src/k8s.io/cloud-provider:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/gce:go_default_library",
        "//staging/src/k8s.io/metrics/pkg/client/clientset/versioned/scheme:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
        "//pkg/controller/nodeipam/ipam/cidrset:all-srcs",
        "//pkg/controller/nodeipam/ipam/sync:all-srcs",
        "//pkg/controller/nodeipam/ipam/test:all-srcs",
    ],
    tags = ["automanaged"],
)
