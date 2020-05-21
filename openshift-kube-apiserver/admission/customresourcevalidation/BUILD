load("@io_bazel_rules_go//go:def.bzl", "go_library", "go_test")

go_library(
    name = "go_default_library",
    srcs = [
        "attributes.go",
        "customresourcevalidator.go",
        "helpers.go",
    ],
    importpath = "k8s.io/kubernetes/openshift-kube-apiserver/admission/customresourcevalidation",
    visibility = ["//visibility:public"],
    deps = [
        "//pkg/apis/core/validation:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//vendor/github.com/openshift/api/authorization/v1:go_default_library",
        "//vendor/github.com/openshift/api/config/v1:go_default_library",
        "//vendor/github.com/openshift/api/quota/v1:go_default_library",
        "//vendor/github.com/openshift/api/security/v1:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["customresourcevalidator_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/admission:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//openshift-kube-apiserver/admission/customresourcevalidation/apiserver:all-srcs",
        "//openshift-kube-apiserver/admission/customresourcevalidation/authentication:all-srcs",
        "//openshift-kube-apiserver/admission/customresourcevalidation/clusterresourcequota:all-srcs",
        "//openshift-kube-apiserver/admission/customresourcevalidation/config:all-srcs",
        "//openshift-kube-apiserver/admission/customresourcevalidation/console:all-srcs",
        "//openshift-kube-apiserver/admission/customresourcevalidation/customresourcevalidationregistration:all-srcs",
        "//openshift-kube-apiserver/admission/customresourcevalidation/features:all-srcs",
        "//openshift-kube-apiserver/admission/customresourcevalidation/image:all-srcs",
        "//openshift-kube-apiserver/admission/customresourcevalidation/oauth:all-srcs",
        "//openshift-kube-apiserver/admission/customresourcevalidation/project:all-srcs",
        "//openshift-kube-apiserver/admission/customresourcevalidation/rolebindingrestriction:all-srcs",
        "//openshift-kube-apiserver/admission/customresourcevalidation/scheduler:all-srcs",
        "//openshift-kube-apiserver/admission/customresourcevalidation/securitycontextconstraints:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
