load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "example_handler.go",
        "example_plugin.go",
        "plugin_watcher.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/pluginmanager/pluginwatcher",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/kubelet/pluginmanager/cache:go_default_library",
        "//pkg/kubelet/pluginmanager/pluginwatcher/example_plugin_apis/v1beta1:go_default_library",
        "//pkg/kubelet/pluginmanager/pluginwatcher/example_plugin_apis/v1beta2:go_default_library",
        "//pkg/kubelet/util:go_default_library",
        "//pkg/util/filesystem:go_default_library",
        "//staging/src/k8s.io/kubelet/pkg/apis/pluginregistration/v1:go_default_library",
        "//vendor/github.com/fsnotify/fsnotify:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["plugin_watcher_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//pkg/kubelet/pluginmanager/cache:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/kubelet/pkg/apis/pluginregistration/v1:go_default_library",
        "//vendor/github.com/stretchr/testify/require:go_default_library",
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
        "//pkg/kubelet/pluginmanager/pluginwatcher/example_plugin_apis/v1beta1:all-srcs",
        "//pkg/kubelet/pluginmanager/pluginwatcher/example_plugin_apis/v1beta2:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
