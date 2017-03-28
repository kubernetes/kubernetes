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
        "deployment_controller.go",
        "progress.go",
        "recreate.go",
        "rollback.go",
        "rolling.go",
        "sync.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/core/v1:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions/extensions/v1beta1:go_default_library",
        "//pkg/client/listers/core/v1:go_default_library",
        "//pkg/client/listers/extensions/v1beta1:go_default_library",
        "//pkg/controller:go_default_library",
        "//pkg/controller/deployment/util:go_default_library",
        "//pkg/util/labels:go_default_library",
        "//pkg/util/metrics:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/util/errors",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/client-go/kubernetes/typed/core/v1",
        "//vendor:k8s.io/client-go/pkg/api/v1",
        "//vendor:k8s.io/client-go/tools/cache",
        "//vendor:k8s.io/client-go/tools/record",
        "//vendor:k8s.io/client-go/util/integer",
        "//vendor:k8s.io/client-go/util/workqueue",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "deployment_controller_test.go",
        "progress_test.go",
        "recreate_test.go",
        "rolling_test.go",
        "sync_test.go",
    ],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/clientset/fake:go_default_library",
        "//pkg/client/informers/informers_generated/externalversions:go_default_library",
        "//pkg/controller:go_default_library",
        "//pkg/controller/deployment/util:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/intstr",
        "//vendor:k8s.io/apimachinery/pkg/util/uuid",
        "//vendor:k8s.io/client-go/testing",
        "//vendor:k8s.io/client-go/tools/record",
        "//vendor:k8s.io/client-go/util/workqueue",
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
        "//pkg/controller/deployment/util:all-srcs",
    ],
    tags = ["automanaged"],
)
