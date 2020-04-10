load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = ["controller.go"],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kube-aggregator/pkg/controllers/openapi",
    importpath = "k8s.io/kube-aggregator/pkg/controllers/openapi",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/client-go/util/workqueue:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/apis/apiregistration/v1:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/controllers/openapi/aggregator:go_default_library",
        "//vendor/k8s.io/klog/v2:go_default_library",
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
        "//staging/src/k8s.io/kube-aggregator/pkg/controllers/openapi/aggregator:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
