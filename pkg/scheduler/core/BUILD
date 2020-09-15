load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "extender.go",
        "generic_scheduler.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/scheduler/core",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/api/v1/pod:go_default_library",
        "//pkg/scheduler/apis/config:go_default_library",
        "//pkg/scheduler/framework/runtime:go_default_library",
        "//pkg/scheduler/framework/v1alpha1:go_default_library",
        "//pkg/scheduler/internal/cache:go_default_library",
        "//pkg/scheduler/internal/parallelize:go_default_library",
        "//pkg/scheduler/metrics:go_default_library",
        "//pkg/scheduler/profile:go_default_library",
        "//pkg/scheduler/util:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/net:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/kube-scheduler/extender/v1:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
        "//vendor/k8s.io/utils/trace:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "extender_test.go",
        "generic_scheduler_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//pkg/controller/volume/persistentvolume/util:go_default_library",
        "//pkg/scheduler/apis/config:go_default_library",
        "//pkg/scheduler/framework/plugins/defaultbinder:go_default_library",
        "//pkg/scheduler/framework/plugins/noderesources:go_default_library",
        "//pkg/scheduler/framework/plugins/podtopologyspread:go_default_library",
        "//pkg/scheduler/framework/plugins/queuesort:go_default_library",
        "//pkg/scheduler/framework/plugins/selectorspread:go_default_library",
        "//pkg/scheduler/framework/plugins/volumebinding:go_default_library",
        "//pkg/scheduler/framework/runtime:go_default_library",
        "//pkg/scheduler/framework/v1alpha1:go_default_library",
        "//pkg/scheduler/internal/cache:go_default_library",
        "//pkg/scheduler/internal/queue:go_default_library",
        "//pkg/scheduler/profile:go_default_library",
        "//pkg/scheduler/testing:go_default_library",
        "//pkg/scheduler/util:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/fake:go_default_library",
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
