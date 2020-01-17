load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "factory.go",
        "generic.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/code-generator/_examples/apiserver/informers/externalversions",
    importpath = "k8s.io/code-generator/_examples/apiserver/informers/externalversions",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/apiserver/apis/example/v1:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/apiserver/apis/example2/v1:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/apiserver/apis/example3.io/v1:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/apiserver/clientset/versioned:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/apiserver/informers/externalversions/example:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/apiserver/informers/externalversions/example2:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/apiserver/informers/externalversions/example3.io:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/apiserver/informers/externalversions/internalinterfaces:go_default_library",
    ],
)
