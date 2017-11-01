package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "device_plugin_handler.go",
        "device_plugin_handler_stub.go",
        "device_plugin_stub.go",
        "endpoint.go",
        "manager.go",
        "pod_devices.go",
        "types.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/cm/deviceplugin",
    deps = [
        "//pkg/api/v1/helper:go_default_library",
        "//pkg/kubelet/apis/deviceplugin/v1alpha:go_default_library",
        "//pkg/kubelet/container:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/golang.org/x/net/context:go_default_library",
        "//vendor/google.golang.org/grpc:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
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
)

go_test(
    name = "go_default_test",
    srcs = [
        "device_plugin_handler_test.go",
        "endpoint_test.go",
        "manager_test.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/kubelet/cm/deviceplugin",
    library = ":go_default_library",
    deps = [
        "//pkg/kubelet/apis/deviceplugin/v1alpha:go_default_library",
        "//vendor/github.com/stretchr/testify/assert:go_default_library",
        "//vendor/github.com/stretchr/testify/require:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/uuid:go_default_library",
    ],
)
