load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "enablement.go",
        "intialization.go",
    ],
    importpath = "k8s.io/kubernetes/openshift-kube-apiserver/enablement",
    visibility = ["//visibility:public"],
    deps = [
        "//openshift-kube-apiserver/configdefault:go_default_library",
        "//pkg/capabilities:go_default_library",
        "//pkg/kubelet/types:go_default_library",
        "//plugin/pkg/auth/authorizer/rbac/bootstrappolicy:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/tools/clientcmd/api:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/apiserver:go_default_library",
        "//vendor/github.com/openshift/api/config/v1:go_default_library",
        "//vendor/github.com/openshift/api/kubecontrolplane/v1:go_default_library",
        "//vendor/github.com/openshift/api/osin/v1:go_default_library",
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
