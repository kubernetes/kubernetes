package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "available_controller.go",
        "metrics.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kube-aggregator/pkg/controllers/status",
    importpath = "k8s.io/kube-aggregator/pkg/controllers/status",
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/api/meta:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/client-go/informers/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/listers/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/client-go/util/workqueue:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/apis/apiregistration:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/client/clientset_generated/internalclientset/typed/apiregistration/internalversion:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/client/informers/internalversion/apiregistration/internalversion:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/client/listers/apiregistration/internalversion:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/controllers:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["available_controller_test.go"],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/client-go/listers/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/testing:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/apis/apiregistration:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/client/clientset_generated/internalclientset/fake:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/client/listers/apiregistration/internalversion:go_default_library",
        "//vendor/github.com/davecgh/go-spew/spew:go_default_library",
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
)
