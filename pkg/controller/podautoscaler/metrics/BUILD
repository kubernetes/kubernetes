package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "metrics_client.go",
        "utilization.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/core/internalversion:go_default_library",
        "//pkg/labels:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/heapster/metrics/api/v1/types",
        "//vendor:k8s.io/heapster/metrics/apis/metrics/v1alpha1",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["metrics_client_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/fake:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/testing/core:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/runtime:go_default_library",
        "//vendor:github.com/stretchr/testify/assert",
        "//vendor:k8s.io/heapster/metrics/api/v1/types",
        "//vendor:k8s.io/heapster/metrics/apis/metrics/v1alpha1",
    ],
)
