load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "admission.go",
        "doc.go",
    ],
    importpath = "k8s.io/kubernetes/openshift-kube-apiserver/admission/autoscaling/clusterresourceoverride",
    visibility = ["//visibility:public"],
    deps = [
        "//openshift-kube-apiserver/admission/autoscaling/apis/clusterresourceoverride:go_default_library",
        "//openshift-kube-apiserver/admission/autoscaling/apis/clusterresourceoverride/v1:go_default_library",
        "//openshift-kube-apiserver/admission/autoscaling/apis/clusterresourceoverride/validation:go_default_library",
        "//pkg/apis/core:go_default_library",
        "//plugin/pkg/admission/limitranger:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission/initializer:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes:go_default_library",
        "//staging/src/k8s.io/client-go/listers/core/v1:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/config/helpers:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["admission_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//openshift-kube-apiserver/admission/autoscaling/apis/clusterresourceoverride:go_default_library",
        "//openshift-kube-apiserver/admission/autoscaling/apis/clusterresourceoverride/v1:go_default_library",
        "//openshift-kube-apiserver/admission/autoscaling/apis/clusterresourceoverride/validation:go_default_library",
        "//pkg/apis/core:go_default_library",
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//staging/src/k8s.io/client-go/listers/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/config/helpers:go_default_library",
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
