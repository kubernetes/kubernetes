load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = ["plugin_manager.go"],
    importpath = "k8s.io/kubernetes/pkg/kubelet/pluginmanager",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubelet/config:go_default_library",
        "//pkg/kubelet/pluginmanager/cache:go_default_library",
        "//pkg/kubelet/pluginmanager/metrics:go_default_library",
        "//pkg/kubelet/pluginmanager/operationexecutor:go_default_library",
        "//pkg/kubelet/pluginmanager/pluginwatcher:go_default_library",
        "//pkg/kubelet/pluginmanager/reconciler:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/client-go/tools/record:go_default_library",
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
        "//pkg/kubelet/pluginmanager/cache:all-srcs",
        "//pkg/kubelet/pluginmanager/metrics:all-srcs",
        "//pkg/kubelet/pluginmanager/operationexecutor:all-srcs",
        "//pkg/kubelet/pluginmanager/pluginwatcher:all-srcs",
        "//pkg/kubelet/pluginmanager/reconciler:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)

go_test(
    name = "go_default_test",
    srcs = ["plugin_manager_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/kubelet/apis/pluginregistration/v1:go_default_library",
        "//pkg/kubelet/config:go_default_library",
        "//pkg/kubelet/pluginmanager/cache:go_default_library",
        "//pkg/kubelet/pluginmanager/pluginwatcher:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/client-go/tools/record:go_default_library",
        "//vendor/github.com/stretchr/testify/require:go_default_library",
    ],
)
