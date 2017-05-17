package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "resources.go",
        "start.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/pborman/uuid:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1/unstructured:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/authorization/authorizerfactory:go_default_library",
        "//vendor/k8s.io/apiserver/pkg/server:go_default_library",
        "//vendor/k8s.io/client-go/dynamic:go_default_library",
        "//vendor/k8s.io/kube-apiextensions-server/pkg/apis/apiextensions/v1alpha1:go_default_library",
        "//vendor/k8s.io/kube-apiextensions-server/pkg/apiserver:go_default_library",
        "//vendor/k8s.io/kube-apiextensions-server/pkg/client/clientset/clientset:go_default_library",
        "//vendor/k8s.io/kube-apiextensions-server/pkg/cmd/server:go_default_library",
    ],
)
