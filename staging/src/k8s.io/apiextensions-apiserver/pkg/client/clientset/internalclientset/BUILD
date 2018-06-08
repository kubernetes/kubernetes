package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "clientset.go",
        "doc.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiextensions-apiserver/pkg/client/clientset/internalclientset",
    importpath = "k8s.io/apiextensions-apiserver/pkg/client/clientset/internalclientset",
    deps = [
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/client/clientset/internalclientset/typed/apiextensions/internalversion:go_default_library",
        "//staging/src/k8s.io/client-go/discovery:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/util/flowcontrol:go_default_library",
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
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/client/clientset/internalclientset/fake:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/client/clientset/internalclientset/scheme:all-srcs",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/client/clientset/internalclientset/typed/apiextensions/internalversion:all-srcs",
    ],
    tags = ["automanaged"],
)
