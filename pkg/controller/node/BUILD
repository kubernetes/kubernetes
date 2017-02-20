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
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/core/v1:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/extensions/v1beta1:go_default_library",
        "//pkg/client/listers/core/v1:go_default_library",
        "//pkg/client/listers/extensions/v1beta1:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/kubelet/util/format:go_default_library",
        "//pkg/util/metrics:go_default_library",
        "//pkg/util/node:go_default_library",
        "//pkg/util/system:go_default_library",
        "//pkg/util/version:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
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
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
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
