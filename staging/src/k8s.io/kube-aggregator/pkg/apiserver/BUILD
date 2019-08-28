package(default_visibility = ["//visibility:public"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = [
        "handler_apis_test.go",
        "handler_proxy_test.go",
    ],
    embed = [":go_default_library"],
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/equality:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/diff:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/proxy:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/authentication/user:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/endpoints/request:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/apis/apiregistration/v1:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/apiserver/scheme:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/client/listers/apiregistration/v1:go_default_library",
        "//vendor/golang.org/x/net/websocket:go_default_library",
        "//vendor/k8s.io/utils/pointer:go_default_library",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "apiserver.go",
        "apiservice_controller.go",
        "handler_apis.go",
        "handler_proxy.go",
        "resolvers.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/k8s.io/kube-aggregator/pkg/apiserver",
    importpath = "k8s.io/kube-aggregator/pkg/apiserver",
    deps = [
        "//staging/src/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/schema:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/runtime/serializer:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/httpstream:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/httpstream/spdy:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/net:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/proxy:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/runtime:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/sets:go_default_library",
        "//staging/src/k8s.io/apimachinery/pkg/util/wait:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/endpoints/handlers/negotiation:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/endpoints/handlers/responsewriters:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/endpoints/metrics:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/endpoints/request:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/features:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/server/storage:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/feature:go_default_library",
        "//staging/src/k8s.io/apiserver/pkg/util/proxy:go_default_library",
        "//staging/src/k8s.io/client-go/listers/core/v1:go_default_library",
        "//staging/src/k8s.io/client-go/pkg/version:go_default_library",
        "//staging/src/k8s.io/client-go/rest:go_default_library",
        "//staging/src/k8s.io/client-go/tools/cache:go_default_library",
        "//staging/src/k8s.io/client-go/transport:go_default_library",
        "//staging/src/k8s.io/client-go/util/workqueue:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/apis/apiregistration/v1:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/apis/apiregistration/v1/helper:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/apis/apiregistration/v1beta1:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/apiserver/scheme:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/client/clientset_generated/clientset:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/client/informers/externalversions:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/client/informers/externalversions/apiregistration/v1:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/client/listers/apiregistration/v1:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/controllers:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/controllers/openapi:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/controllers/openapi/aggregator:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/controllers/status:go_default_library",
        "//staging/src/k8s.io/kube-aggregator/pkg/registry/apiservice/rest:go_default_library",
        "//vendor/k8s.io/klog:go_default_library",
        "//vendor/k8s.io/kube-openapi/pkg/common:go_default_library",
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
        "//staging/src/k8s.io/kube-aggregator/pkg/apiserver/scheme:all-srcs",
    ],
    tags = ["automanaged"],
)
