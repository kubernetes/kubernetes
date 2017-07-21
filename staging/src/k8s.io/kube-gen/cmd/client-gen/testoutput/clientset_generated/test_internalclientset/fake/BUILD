package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "clientset_generated.go",
        "doc.go",
        "register.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//vendor/k8s.io/client-go/discovery:go_default_library",
        "//vendor/k8s.io/client-go/discovery/fake:go_default_library",
        "//vendor/k8s.io/client-go/testing:go_default_library",
        "//vendor/k8s.io/kube-gen/cmd/client-gen/test_apis/testgroup:go_default_library",
        "//vendor/k8s.io/kube-gen/cmd/client-gen/testoutput/clientset_generated/test_internalclientset:go_default_library",
        "//vendor/k8s.io/kube-gen/cmd/client-gen/testoutput/clientset_generated/test_internalclientset/typed/testgroup/internalversion:go_default_library",
        "//vendor/k8s.io/kube-gen/cmd/client-gen/testoutput/clientset_generated/test_internalclientset/typed/testgroup/internalversion/fake:go_default_library",
    ],
)
