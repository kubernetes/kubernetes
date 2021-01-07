package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["node_ipam_controller_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/fake:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/gce:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/gce/nodeipam/ipam:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/gce/nodeipam/util:go_default_library",
        "//vendor/k8s.io/utils/net:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "legacyprovider.go",
        "node_ipam_controller.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/legacy-cloud-providers/gce/nodeipam",
    importpath = "k8s.io/legacy-cloud-providers/gce/nodeipam",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/client-go/informers/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/typed/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/listers/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        "//staging/src/k8s.io/cloud-provider:go_default_library",
        "//staging/src/k8s.io/component-base/metrics/prometheus/ratelimiter:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/gce/nodeipam/ipam:go_default_library",
        "//staging/src/k8s.io/legacy-cloud-providers/gce/nodeipam/ipam/sync:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//staging/src/k8s.io/legacy-cloud-providers/gce/nodeipam/config:all-srcs",
        "//staging/src/k8s.io/legacy-cloud-providers/gce/nodeipam/ipam:all-srcs",
        "//staging/src/k8s.io/legacy-cloud-providers/gce/nodeipam/util:all-srcs",
    ],
    tags = ["automanaged"],
)
