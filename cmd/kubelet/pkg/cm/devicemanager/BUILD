load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "device_plugin_stub.go",
        "endpoint.go",
        "manager.go",
        "manager_stub.go",
        "pod_devices.go",
        "types.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/kubelet/pkg/cm/devicemanager",
    visibility = ["//visibility:public"],
    deps = [
        "//cmd/kubelet/pkg/apis/deviceplugin/v1beta1:go_default_library",
        "//cmd/kubelet/pkg/apis/pluginregistration/v1:go_default_library",
        "//cmd/kubelet/pkg/apis/podresources/v1alpha1:go_default_library",
        "//cmd/kubelet/pkg/checkpointmanager:go_default_library",
        "//cmd/kubelet/pkg/checkpointmanager/errors:go_default_library",
        "//cmd/kubelet/pkg/cm/devicemanager/checkpoint:go_default_library",
        "//cmd/kubelet/pkg/config:go_default_library",
        "//cmd/kubelet/pkg/container:go_default_library",
        "//cmd/kubelet/pkg/lifecycle:go_default_library",
        "//cmd/kubelet/pkg/metrics:go_default_library",
        "//cmd/kubelet/pkg/util/pluginwatcher:go_default_library",
        "//pkg/apis/core/v1/helper:go_default_library",
        "//pkg/scheduler/nodeinfo:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "endpoint_test.go",
        "manager_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//cmd/kubelet/pkg/apis/deviceplugin/v1beta1:go_default_library",
        "//cmd/kubelet/pkg/apis/pluginregistration/v1:go_default_library",
        "//cmd/kubelet/pkg/checkpointmanager:go_default_library",
        "//cmd/kubelet/pkg/lifecycle:go_default_library",
        "//cmd/kubelet/pkg/util/pluginwatcher:go_default_library",
        "//pkg/scheduler/nodeinfo:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/github.com/stretchr/testify/require:go_default_library",
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
        "//cmd/kubelet/pkg/cm/devicemanager/checkpoint:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
