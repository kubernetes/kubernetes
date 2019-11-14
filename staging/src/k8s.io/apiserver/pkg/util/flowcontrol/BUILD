load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "reqmgmt.go",
        "reqmgmt-config.go",
        "rule.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/apiserver/pkg/util/flowcontrol",
    importpath = "k8s.io/apiserver/pkg/util/flowcontrol",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/flowcontrol/v1alpha1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/clock:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/serviceaccount:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/endpoints/request:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/apihelpers:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/bootstrap:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/counter:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing/queueset:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/metrics:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/shufflesharding:go_default_library",
        "//staging/src/k8s.io/client-go/informers:go_default_library",
        "//staging/src/k8s.io/client-go/kubernetes/typed/flowcontrol/v1alpha1:go_default_library",
        "//staging/src/k8s.io/client-go/listers/flowcontrol/v1alpha1:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/client-go/util/workqueue:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
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
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/bootstrap:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/counter:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/fairqueuing:all-srcs",
        "//staging/src/k8s.io/apiserver/pkg/util/flowcontrol/metrics:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
