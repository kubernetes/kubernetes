load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["establishing_controller.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiextensions-apiserver/pkg/controller/establish",
    importpath = "k8s.io/apiextensions-apiserver/pkg/controller/establish",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/apis/apiextensions:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/client/clientset/internalclientset/typed/apiextensions/internalversion:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/client/informers/internalversion/apiextensions/internalversion:go_default_library",
        "//staging/src/k8s.io/apiextensions-apiserver/pkg/client/listers/apiextensions/internalversion:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/client-go/util/workqueue:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
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
    visibility = ["//visibility:public"],
)
