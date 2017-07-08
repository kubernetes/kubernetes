package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = ["storage_rbac.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apis/rbac:go_default_library",
        "//pkg/apis/rbac/v1alpha1:go_default_library",
        "//pkg/apis/rbac/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/core/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion:go_default_library",
        "//pkg/client/retry:go_default_library",
        "//pkg/registry/rbac/clusterrole:go_default_library",
        "//pkg/registry/rbac/clusterrole/policybased:go_default_library",
        "//pkg/registry/rbac/clusterrole/storage:go_default_library",
        "//pkg/registry/rbac/clusterrolebinding:go_default_library",
        "//pkg/registry/rbac/clusterrolebinding/policybased:go_default_library",
        "//pkg/registry/rbac/clusterrolebinding/storage:go_default_library",
        "//pkg/registry/rbac/reconciliation:go_default_library",
        "//pkg/registry/rbac/role:go_default_library",
        "//pkg/registry/rbac/role/policybased:go_default_library",
        "//pkg/registry/rbac/role/storage:go_default_library",
        "//pkg/registry/rbac/rolebinding:go_default_library",
        "//pkg/registry/rbac/rolebinding/policybased:go_default_library",
        "//pkg/registry/rbac/rolebinding/storage:go_default_library",
        "//pkg/registry/rbac/validation:go_default_library",
        "//plugin/pkg/auth/authorizer/rbac/bootstrappolicy:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authorization/authorizer:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/registry/generic:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/registry/rest:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/server:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/server/storage:go_default_library",
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
)
