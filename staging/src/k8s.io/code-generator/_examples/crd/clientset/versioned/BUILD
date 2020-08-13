load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "clientset.go",
        "doc.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/code-generator/_examples/crd/clientset/versioned",
    importpath = "k8s.io/code-generator/_examples/crd/clientset/versioned",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/client-go/discovery:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/util/flowcontrol:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/crd/clientset/versioned/typed/example/v1:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/crd/clientset/versioned/typed/example2/v1:go_default_library",
    ],
)
