load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["cr_validation_registration.go"],
    importpath = "k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/customresourcevalidationregistration",
    visibility = ["//visibility:public"],
    deps = [
        "//openshift-kube-apiserver/admission/customresourcevalidation/apiserver:go_default_library",
        "//openshift-kube-apiserver/admission/customresourcevalidation/authentication:go_default_library",
        "//openshift-kube-apiserver/admission/customresourcevalidation/clusterresourcequota:go_default_library",
        "//openshift-kube-apiserver/admission/customresourcevalidation/config:go_default_library",
        "//openshift-kube-apiserver/admission/customresourcevalidation/console:go_default_library",
        "//openshift-kube-apiserver/admission/customresourcevalidation/features:go_default_library",
        "//openshift-kube-apiserver/admission/customresourcevalidation/image:go_default_library",
        "//openshift-kube-apiserver/admission/customresourcevalidation/oauth:go_default_library",
        "//openshift-kube-apiserver/admission/customresourcevalidation/project:go_default_library",
        "//openshift-kube-apiserver/admission/customresourcevalidation/rolebindingrestriction:go_default_library",
        "//openshift-kube-apiserver/admission/customresourcevalidation/scheduler:go_default_library",
        "//openshift-kube-apiserver/admission/customresourcevalidation/securitycontextconstraints:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
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
