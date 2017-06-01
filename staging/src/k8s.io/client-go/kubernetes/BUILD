package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "clientset.go",
        "doc.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/client-go/discovery:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/apps/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/authentication/v1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/authentication/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/authorization/v1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/authorization/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/autoscaling/v1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/autoscaling/v2alpha1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/batch/v1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/batch/v2alpha1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/certificates/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/core/v1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/extensions/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/policy/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/rbac/v1alpha1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/rbac/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/settings/v1alpha1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/storage/v1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/storage/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/rest:go_default_library",
        "//vendor/k8s.io/client-go/util/flowcontrol:go_default_library",
    ],
)
