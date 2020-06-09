load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "fake_apiregistration_client.go",
        "fake_apiservice.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kube-aggregator/pkg/client/clientset_generated/deprecated/typed/apiregistration/v1beta1/fake",
    importpath = "k8s.io/kube-aggregator/pkg/client/clientset_generated/deprecated/typed/apiregistration/v1beta1/fake",
    visibility = ["//staging/src/k8s.io/kube-aggregator/pkg/client/clientset_generated/deprecated:__subpackages__"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/types:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/testing:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/apis/apiregistration/v1beta1:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/client/clientset_generated/deprecated/typed/apiregistration/v1beta1:go_default_library",
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
