load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "kubecontrolplane_default.go",
        "kubecontrolplane_refs.go",
    ],
    importpath = "k8s.io/kubernetes/openshift-kube-apiserver/configdefault",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/github.com/openshift/api/kubecontrolplane/v1:go_default_library",
        "//vendor/github.com/openshift/api/osin/v1:go_default_library",
        "//vendor/github.com/openshift/library-go/pkg/config/configdefaults:go_default_library",
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
