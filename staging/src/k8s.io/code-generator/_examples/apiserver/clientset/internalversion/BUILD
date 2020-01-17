load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "clientset.go",
        "doc.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/code-generator/_examples/apiserver/clientset/internalversion",
    importpath = "k8s.io/code-generator/_examples/apiserver/clientset/internalversion",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/client-go/discovery:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/util/flowcontrol:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/apiserver/clientset/internalversion/typed/example/internalversion:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/apiserver/clientset/internalversion/typed/example2/internalversion:go_default_library",
        "//staging/src/k8s.io/code-generator/_examples/apiserver/clientset/internalversion/typed/example3.io/internalversion:go_default_library",
    ],
)
