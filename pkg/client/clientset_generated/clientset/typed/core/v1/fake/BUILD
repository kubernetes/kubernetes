package(default_visibility = ["//visibility:public"])

licenses(["notice"])

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
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/policy/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/core/v1:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/testing/core:go_default_library",
        "//pkg/fields:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/watch",
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
