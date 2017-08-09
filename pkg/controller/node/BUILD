package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["nodecontroller_test.go"],
    library = ":go_default_library",
    deps = [
        "//pkg/cloudprovider:go_default_library",
        "//pkg/cloudprovider/providers/fake:go_default_library",
        "//pkg/controller:go_default_library",
        "//pkg/controller/node/ipam:go_default_library",
        "//pkg/controller/node/scheduler:go_default_library",
        "//pkg/controller/node/util:go_default_library",
        "//pkg/controller/testutil:go_default_library",
        "//pkg/kubelet/apis:go_default_library",
        "//pkg/util/node:go_default_library",
        "//pkg/util/taints:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/api/extensions/v1beta1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/resource:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/client-go/informers:go_default_library",
        "//vendor/k8s.io/client-go/informers/core/v1:go_default_library",
        "//vendor/k8s.io/client-go/informers/extensions/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/fake:go_default_library",
        "//vendor/k8s.io/client-go/testing:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "metrics.go",
        "node_controller.go",
    ],
    deps = [
        "//pkg/api/v1/node:go_default_library",
        "//pkg/cloudprovider:go_default_library",
        "//pkg/controller:go_default_library",
        "//pkg/controller/node/ipam:go_default_library",
        "//pkg/controller/node/scheduler:go_default_library",
        "//pkg/controller/node/util:go_default_library",
        "//pkg/util/metrics:go_default_library",
        "//pkg/util/node:go_default_library",
        "//pkg/util/system:go_default_library",
        "//pkg/util/taints:go_default_library",
        "//pkg/util/version:go_default_library",
        "//plugin/pkg/scheduler/algorithm:go_default_library",
        "//vendor/github.com/golang/glog:go_default_library",
        "//vendor/github.com/prometheus/client_golang/prometheus:go_default_library",
        "//vendor/k8s.io/api/core/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/fields:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/types:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//vendor/k8s.io/client-go/informers/core/v1:go_default_library",
        "//vendor/k8s.io/client-go/informers/extensions/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/scheme:go_default_library",
        "//vendor/k8s.io/client-go/kubernetes/typed/core/v1:go_default_library",
        "//vendor/k8s.io/client-go/listers/core/v1:go_default_library",
        "//vendor/k8s.io/client-go/listers/extensions/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
        "//vendor/k8s.io/client-go/tools/record:go_default_library",
        "//vendor/k8s.io/client-go/util/flowcontrol:go_default_library",
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
        "//pkg/controller/node/ipam:all-srcs",
        "//pkg/controller/node/scheduler:all-srcs",
        "//pkg/controller/node/util:all-srcs",
    ],
    tags = ["automanaged"],
)
