package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "fake_componentstatus.go",
        "fake_configmap.go",
        "fake_core_client.go",
        "fake_endpoints.go",
        "fake_event.go",
        "fake_event_expansion.go",
        "fake_limitrange.go",
        "fake_namespace.go",
        "fake_namespace_expansion.go",
        "fake_node.go",
        "fake_node_expansion.go",
        "fake_persistentvolume.go",
        "fake_persistentvolumeclaim.go",
        "fake_pod.go",
        "fake_pod_expansion.go",
        "fake_podtemplate.go",
        "fake_replicationcontroller.go",
        "fake_resourcequota.go",
        "fake_secret.go",
        "fake_service.go",
        "fake_service_expansion.go",
        "fake_serviceaccount.go",
    ],
    deps = [
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/api/extensions/v1beta1:go_default_library",
        "//vendor/k8s.io/api/policy/v1beta1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/core/v1:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
        "//vendor/k8s.io/client-go/testing:go_default_library",
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
