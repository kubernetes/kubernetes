load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "cycle_state.go",
        "extender.go",
        "framework.go",
        "interface.go",
        "listers.go",
        "metrics_recorder.go",
        "registry.go",
        "types.go",
        "waiting_pods_map.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/scheduler/framework/v1alpha1",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/apis/core/v1/helper:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/scheduler/apis/config:go_default_library",
        "//pkg/scheduler/apis/config/scheme:go_default_library",
        "//pkg/scheduler/internal/parallelize:go_default_library",
        "//pkg/scheduler/metrics:go_default_library",
        "//pkg/scheduler/util:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/json:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/component-base/metrics:go_default_library",
        "//staging/src/k8s.io/kube-scheduler/config/v1alpha2:go_default_library",
        "//staging/src/k8s.io/kube-scheduler/extender/v1:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
        "//vendor/sigs.k8s.io/yaml:go_default_library",
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
        "//pkg/scheduler/framework/v1alpha1/fake:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = [
        "cycle_state_test.go",
        "framework_test.go",
        "interface_test.go",
        "registry_test.go",
        "types_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/scheduler/apis/config:go_default_library",
        "//pkg/scheduler/metrics:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/github.com/google/go-cmp/cmp:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/github.com/prometheus/client_model/go:go_default_library",
    ],
)
