load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["default_preemption.go"],
    importpath = "k8s.io/kubernetes/pkg/scheduler/framework/plugins/defaultpreemption",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/api/v1/pod:go_default_library",
        "//pkg/features:go_default_library",
        "//pkg/scheduler/core:go_default_library",
        "//pkg/scheduler/framework/v1alpha1:go_default_library",
        "//pkg/scheduler/internal/parallelize:go_default_library",
        "//pkg/scheduler/metrics:go_default_library",
        "//pkg/scheduler/util:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/policy/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/kube-scheduler/extender/v1:go_default_library",
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
    srcs = ["default_preemption_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/controller/volume/scheduling:go_default_library",
        "//pkg/scheduler/framework/plugins/defaultbinder:go_default_library",
        "//pkg/scheduler/framework/plugins/interpodaffinity:go_default_library",
        "//pkg/scheduler/framework/plugins/nodeaffinity:go_default_library",
        "//pkg/scheduler/framework/plugins/nodelabel:go_default_library",
        "//pkg/scheduler/framework/plugins/nodename:go_default_library",
        "//pkg/scheduler/framework/plugins/noderesources:go_default_library",
        "//pkg/scheduler/framework/plugins/nodeunschedulable:go_default_library",
        "//pkg/scheduler/framework/plugins/podtopologyspread:go_default_library",
        "//pkg/scheduler/framework/plugins/queuesort:go_default_library",
        "//pkg/scheduler/framework/plugins/tainttoleration:go_default_library",
        "//pkg/scheduler/framework/plugins/volumerestrictions:go_default_library",
        "//pkg/scheduler/framework/plugins/volumezone:go_default_library",
        "//pkg/scheduler/framework/runtime:go_default_library",
        "//pkg/scheduler/framework/v1alpha1:go_default_library",
        "//pkg/scheduler/internal/cache:go_default_library",
        "//pkg/scheduler/internal/queue:go_default_library",
        "//pkg/scheduler/testing:go_default_library",
        "//pkg/scheduler/util:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/api/policy/v1beta1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/fake:go_default_library",
        "//staging/src/k8s.io/client-go/testing:go_default_library",
        "//staging/src/k8s.io/client-go/tools/events:go_default_library",
        "//staging/src/k8s.io/kube-scheduler/extender/v1:go_default_library",
    ],
)
