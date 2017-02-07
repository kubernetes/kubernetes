package(default_visibility = ["//visibility:public"])

licenses(["notice"])

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
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//cmd/kube-aggregator/pkg/apis/apiregistration:go_default_library",
        "//cmd/kube-aggregator/pkg/client/listers/apiregistration/internalversion:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/listers/core/v1:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/api/equality",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/diff",
        "//vendor:k8s.io/apimachinery/pkg/util/sets",
        "//vendor:k8s.io/apiserver/pkg/authentication/user",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
        "//vendor:k8s.io/client-go/tools/cache",
    ],
)

go_library(
    name = "go_default_library",
    srcs = [
        "apiserver.go",
        "apiservice_controller.go",
        "handler_apis.go",
        "handler_proxy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/kube-aggregator/pkg/apis/apiregistration:go_default_library",
        "//cmd/kube-aggregator/pkg/apis/apiregistration/v1alpha1:go_default_library",
        "//cmd/kube-aggregator/pkg/client/clientset_generated/clientset:go_default_library",
        "//cmd/kube-aggregator/pkg/client/clientset_generated/internalclientset:go_default_library",
        "//cmd/kube-aggregator/pkg/client/informers:go_default_library",
        "//cmd/kube-aggregator/pkg/client/informers/apiregistration/internalversion:go_default_library",
        "//cmd/kube-aggregator/pkg/client/listers/apiregistration/internalversion:go_default_library",
        "//cmd/kube-aggregator/pkg/registry/apiservice/etcd:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/client/clientset_generated/clientset:go_default_library",
        "//pkg/client/informers/informers_generated:go_default_library",
        "//pkg/client/listers/core/v1:go_default_library",
        "//pkg/controller:go_default_library",
        "//pkg/version:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/api/errors",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/httpstream/spdy",
        "//vendor:k8s.io/apimachinery/pkg/util/runtime",
        "//vendor:k8s.io/apimachinery/pkg/util/wait",
        "//vendor:k8s.io/apiserver/pkg/endpoints/filters",
        "//vendor:k8s.io/apiserver/pkg/endpoints/handlers/responsewriters",
        "//vendor:k8s.io/apiserver/pkg/endpoints/request",
        "//vendor:k8s.io/apiserver/pkg/registry/generic",
        "//vendor:k8s.io/apiserver/pkg/registry/generic/rest",
        "//vendor:k8s.io/apiserver/pkg/registry/rest",
        "//vendor:k8s.io/apiserver/pkg/server",
        "//vendor:k8s.io/apiserver/pkg/server/filters",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/tools/cache",
        "//vendor:k8s.io/client-go/transport",
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
    srcs = [":package-srcs"],
    tags = ["automanaged"],
)
