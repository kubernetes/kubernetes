package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_library(
    name = "go_default_library",
    srcs = [
        "cidr_allocator.go",
        "cidr_set.go",
        "controller_utils.go",
        "doc.go",
        "metrics.go",
        "nodecontroller.go",
        "rate_limited_queue.go",
        "taint_controller.go",
        "timed_workers.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/core/v1:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/extensions/v1beta1:go_default_library",
        "//pkg/client/listers/core/v1:go_default_library",
        "//pkg/client/listers/extensions/v1beta1:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/controller:go_default_library",
        "//pkg/kubelet/util/format:go_default_library",
        "//pkg/util/metrics:go_default_library",
        "//pkg/util/node:go_default_library",
        "//pkg/util/system:go_default_library",
        "//pkg/util/version:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/prometheus/client_golang/prometheus",
        "//vendor:k8s.io/apimachinery/pkg/api/equality",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/fields",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/client-go/kubernetes/typed/core/v1",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/tools/cache",
        "//vendor:k8s.io/client-go/tools/record",
        "//vendor:k8s.io/client-go/util/flowcontrol",
        "//vendor:k8s.io/client-go/util/workqueue",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "cidr_allocator_test.go",
        "cidr_set_test.go",
        "nodecontroller_test.go",
        "rate_limited_queue_test.go",
        "taint_controller_test.go",
        "timed_workers_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/client/clientset_generated/clientset/fake:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/core/v1:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/extensions/v1beta1:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/cloudprovider/providers/fake:go_default_library",
        "//pkg/controller:go_default_library",
        "//pkg/controller/node/testutil:go_default_library",
        "//pkg/util/node:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/equality",
        "//vendor:k8s.io/apimachinery/pkg/api/resource",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/client-go/testing",
        "//vendor:k8s.io/client-go/util/flowcontrol",
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
        "//pkg/controller/node/testutil:all-srcs",
    ],
    tags = ["automanaged"],
)
