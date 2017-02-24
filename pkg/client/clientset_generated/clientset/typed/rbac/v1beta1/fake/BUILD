package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "fake_clusterrole.go",
        "fake_clusterrolebinding.go",
        "fake_rbac_client.go",
        "fake_role.go",
        "fake_rolebinding.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/rbac/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset/typed/rbac/v1beta1:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/testing",
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
