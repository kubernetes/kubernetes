package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["autoregister_controller_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/client-go/testing:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/k8s.io/client-go/util/workqueue:go_default_library",
        "//vendor/k8s.io/kube-aggregator/pkg/apis/apiregistration:go_default_library",
        "//vendor/k8s.io/kube-aggregator/pkg/client/clientset_generated/internalclientset/fake:go_default_library",
        "//vendor/k8s.io/kube-aggregator/pkg/client/listers/apiregistration/internalversion:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = ["autoregister_controller.go"],
    tags = ["automanaged"],
    deps = [
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/conversion:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/k8s.io/client-go/util/workqueue:go_default_library",
        "//vendor/k8s.io/kube-aggregator/pkg/apis/apiregistration:go_default_library",
        "//vendor/k8s.io/kube-aggregator/pkg/client/clientset_generated/internalclientset/typed/apiregistration/internalversion:go_default_library",
        "//vendor/k8s.io/kube-aggregator/pkg/client/informers/internalversion/apiregistration/internalversion:go_default_library",
        "//vendor/k8s.io/kube-aggregator/pkg/client/listers/apiregistration/internalversion:go_default_library",
        "//vendor/k8s.io/kube-aggregator/pkg/controllers:go_default_library",
    ],
)
