load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["endpoint_admission.go"],
    importpath = "k8s.io/kubernetes/openshift-kube-apiserver/admission/network/restrictedendpoints",
    visibility = ["//visibility:public"],
    deps = [
        "//openshift-kube-apiserver/admission/network/apis/restrictedendpoints:go_default_library",
        "//openshift-kube-apiserver/admission/network/apis/restrictedendpoints/v1:go_default_library",
        "//pkg/apis/core:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/initializer:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/config/helpers:go_default_library",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
