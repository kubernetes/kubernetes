load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["validate_scheduler.go"],
    importpath = "k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation/scheduler",
    visibility = ["//visibility:public"],
    deps = [
        "//openshift-kube-apiserver/admission/customresourcevalidation:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/validation:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//vendor/github.com/openshift/api/config/v1:go_default_library",
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
