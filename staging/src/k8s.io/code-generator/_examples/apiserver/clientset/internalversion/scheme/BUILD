load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "register.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/code-generator/_examples/apiserver/clientset/internalversion/scheme",
    importpath = "k8s.io/code-generator/_examples/apiserver/clientset/internalversion/scheme",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/apiserver/apis/example/install:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/apiserver/apis/example2/install:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/apiserver/apis/example3.io/install:go_default_library",
    ],
)
