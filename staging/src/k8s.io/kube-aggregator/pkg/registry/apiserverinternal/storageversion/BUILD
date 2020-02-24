load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "strategy.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kube-aggregator/pkg/registry/apiserverinternal/storageversion",
    importpath = "k8s.io/kube-aggregator/pkg/registry/apiserverinternal/storageversion",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/validation/field:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/apis/apiserverinternal:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/apis/apiserverinternal/validation:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/storage/names:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/apiserver/scheme:go_default_library",
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
        "//staging/src/k8s.io/kube-aggregator/pkg/registry/apiserverinternal/storageversion/storage:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
