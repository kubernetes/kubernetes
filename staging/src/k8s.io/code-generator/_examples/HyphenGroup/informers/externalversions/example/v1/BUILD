load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "clustertesttype.go",
        "interface.go",
        "testtype.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/code-generator/_examples/HyphenGroup/informers/externalversions/example/v1",
    importpath = "k8s.io/code-generator/_examples/HyphenGroup/informers/externalversions/example/v1",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/watch:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/HyphenGroup/apis/example/v1:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/HyphenGroup/clientset/versioned:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/HyphenGroup/informers/externalversions/internalinterfaces:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/HyphenGroup/listers/example/v1:go_default_library",
    ],
)
