package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "categoriesProvider.go",
        "doc.go",
        "endpoint.go",
        "etcd.go",
        "node.go",
        "service.go",
        "shortNamesProvider.go",
        "validate.go",
    ],
    importpath = "k8s.io/kubernetes/pkg/registry/registrytest",
    deps = [
        "//pkg/apis/core:go_default_library",
        "//pkg/kubeapiserver:go_default_library",
        "//pkg/util/slice:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/internalversion:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/endpoints/request:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/registry/generic/registry:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/registry/rest:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/options:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/storage:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/etcd3/testing:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/storagebackend:go_default_library",
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
