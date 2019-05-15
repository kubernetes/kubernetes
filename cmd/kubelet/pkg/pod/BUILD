package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "mirror_client.go",
        "pod_manager.go",
    ],
    importpath = "k8s.io/kubernetes/cmd/kubelet/pkg/pod",
    deps = [
        "//cmd/kubelet/pkg/checkpoint:go_default_library",
        "//cmd/kubelet/pkg/checkpointmanager:go_default_library",
        "//cmd/kubelet/pkg/configmap:go_default_library",
        "//cmd/kubelet/pkg/container:go_default_library",
        "//cmd/kubelet/pkg/secret:go_default_library",
        "//cmd/kubelet/pkg/types:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "mirror_client_test.go",
        "pod_manager_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//cmd/kubelet/pkg/configmap:go_default_library",
        "//cmd/kubelet/pkg/container:go_default_library",
        "//cmd/kubelet/pkg/pod/testing:go_default_library",
        "//cmd/kubelet/pkg/secret:go_default_library",
        "//cmd/kubelet/pkg/types:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
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
        "//cmd/kubelet/pkg/pod/testing:all-srcs",
    ],
    tags = ["automanaged"],
)
