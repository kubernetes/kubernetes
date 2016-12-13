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
        "apiserver.go",
        "apiservice_controller.go",
        "handler_apis.go",
        "handler_proxy.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//cmd/kubernetes-discovery/pkg/apis/apiregistration:go_default_library",
        "//cmd/kubernetes-discovery/pkg/apis/apiregistration/v1alpha1:go_default_library",
        "//cmd/kubernetes-discovery/pkg/client/clientset_generated/internalclientset:go_default_library",
        "//cmd/kubernetes-discovery/pkg/client/clientset_generated/release_1_5:go_default_library",
        "//cmd/kubernetes-discovery/pkg/client/informers:go_default_library",
        "//cmd/kubernetes-discovery/pkg/client/informers/apiregistration/internalversion:go_default_library",
        "//cmd/kubernetes-discovery/pkg/client/listers/apiregistration/internalversion:go_default_library",
        "//cmd/kubernetes-discovery/pkg/registry/apiservice/etcd:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/rest:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/apiserver:go_default_library",
        "//pkg/apiserver/filters:go_default_library",
        "//pkg/auth/handlers:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/client/transport:go_default_library",
        "//pkg/controller:go_default_library",
        "//pkg/genericapiserver:go_default_library",
        "//pkg/genericapiserver/filters:go_default_library",
        "//pkg/labels:go_default_library",
        "//pkg/registry/generic:go_default_library",
        "//pkg/registry/generic/rest:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/httpstream/spdy:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//pkg/util/workqueue:go_default_library",
        "//pkg/version:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = [
        "handler_apis_test.go",
        "handler_proxy_test.go",
    ],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//cmd/kubernetes-discovery/pkg/apis/apiregistration:go_default_library",
        "//cmd/kubernetes-discovery/pkg/client/listers/apiregistration/internalversion:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/apis/meta/v1:go_default_library",
        "//pkg/apiserver/request:go_default_library",
        "//pkg/auth/user:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/diff:go_default_library",
        "//pkg/util/sets:go_default_library",
    ],
)
