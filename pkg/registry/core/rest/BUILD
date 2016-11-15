package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_binary",
    "go_library",
    "go_test",
    "cgo_library",
)

go_library(
    name = "go_default_library",
    srcs = ["storage_core.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/rest:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/apiserver:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/policy/internalversion:go_default_library",
        "//pkg/client/restclient:go_default_library",
        "//pkg/genericapiserver:go_default_library",
        "//pkg/kubelet/client:go_default_library",
        "//pkg/master/ports:go_default_library",
        "//pkg/registry/core/componentstatus:go_default_library",
        "//pkg/registry/core/configmap/etcd:go_default_library",
        "//pkg/registry/core/controller/etcd:go_default_library",
        "//pkg/registry/core/endpoint:go_default_library",
        "//pkg/registry/core/endpoint/etcd:go_default_library",
        "//pkg/registry/core/event/etcd:go_default_library",
        "//pkg/registry/core/limitrange/etcd:go_default_library",
        "//pkg/registry/core/namespace:go_default_library",
        "//pkg/registry/core/namespace/etcd:go_default_library",
        "//pkg/registry/core/node:go_default_library",
        "//pkg/registry/core/node/etcd:go_default_library",
        "//pkg/registry/core/persistentvolume/etcd:go_default_library",
        "//pkg/registry/core/persistentvolumeclaim/etcd:go_default_library",
        "//pkg/registry/core/pod/etcd:go_default_library",
        "//pkg/registry/core/podtemplate/etcd:go_default_library",
        "//pkg/registry/core/rangeallocation:go_default_library",
        "//pkg/registry/core/resourcequota/etcd:go_default_library",
        "//pkg/registry/core/secret/etcd:go_default_library",
        "//pkg/registry/core/service:go_default_library",
        "//pkg/registry/core/service/allocator:go_default_library",
        "//pkg/registry/core/service/allocator/etcd:go_default_library",
        "//pkg/registry/core/service/etcd:go_default_library",
        "//pkg/registry/core/service/ipallocator:go_default_library",
        "//pkg/registry/core/service/portallocator:go_default_library",
        "//pkg/registry/core/serviceaccount/etcd:go_default_library",
        "//pkg/storage/etcd/util:go_default_library",
        "//pkg/util/net:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["storage_core_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api/unversioned:go_default_library",
        "//pkg/storage/storagebackend:go_default_library",
    ],
)
