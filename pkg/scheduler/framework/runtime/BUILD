load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "framework.go",
        "metrics_recorder.go",
        "registry.go",
        "waiting_pods_map.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/scheduler/framework/runtime",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/scheduler/apis/config:go_default_library",
        "//pkg/scheduler/apis/config/scheme:go_default_library",
        "//pkg/scheduler/framework:go_default_library",
        "//pkg/scheduler/internal/parallelize:go_default_library",
        "//pkg/scheduler/metrics:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/json:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/tools/events:go_default_library",
        "//staging/src/k8s.io/component-base/metrics:go_default_library",
        "//staging/src/k8s.io/kube-scheduler/config/v1beta1:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
        "//vendor/sigs.k8s.io/yaml:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "framework_test.go",
        "registry_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/scheduler/apis/config:go_default_library",
        "//pkg/scheduler/framework:go_default_library",
        "//pkg/scheduler/metrics:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/github.com/google/go-cmp/cmp:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/github.com/prometheus/client_model/go:go_default_library",
    ],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

filegroup(
    name = "all-srcs",
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
