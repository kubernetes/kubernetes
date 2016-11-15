package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = ["storage_rbac.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/rest:go_default_library",
        "//pkg/apis/rbac:go_default_library",
        "//pkg/apis/rbac/v1alpha1:go_default_library",
        "//pkg/apis/rbac/validation:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/rbac/internalversion:go_default_library",
        "//pkg/genericapiserver:go_default_library",
        "//pkg/registry/rbac/clusterrole:go_default_library",
        "//pkg/registry/rbac/clusterrole/etcd:go_default_library",
        "//pkg/registry/rbac/clusterrole/policybased:go_default_library",
        "//pkg/registry/rbac/clusterrolebinding:go_default_library",
        "//pkg/registry/rbac/clusterrolebinding/etcd:go_default_library",
        "//pkg/registry/rbac/clusterrolebinding/policybased:go_default_library",
        "//pkg/registry/rbac/role:go_default_library",
        "//pkg/registry/rbac/role/etcd:go_default_library",
        "//pkg/registry/rbac/role/policybased:go_default_library",
        "//pkg/registry/rbac/rolebinding:go_default_library",
        "//pkg/registry/rbac/rolebinding/etcd:go_default_library",
        "//pkg/registry/rbac/rolebinding/policybased:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//plugin/pkg/auth/authorizer/rbac/bootstrappolicy:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)
