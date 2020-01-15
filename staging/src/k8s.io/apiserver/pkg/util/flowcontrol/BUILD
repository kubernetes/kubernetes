load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["filter.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/util/flowcontrol",
    importpath = "k8s.io/apiserver/pkg/util/flowcontrol",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/flowcontrol/v1alpha1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/clock:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/counter:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/queueset:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/filterconfig:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/metrics:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/typed/flowcontrol/v1alpha1:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/counter:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/filterconfig:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/metrics:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
