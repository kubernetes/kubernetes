package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
    "go_test",
)

go_test(
    name = "go_default_test",
    srcs = ["storage_core_test.go"],
    library = ":go_default_library",
    tags = ["automanaged"],
    deps = [
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apiserver/pkg/storage/storagebackend",
    ],
)

go_library(
    name = "go_default_library",
    srcs = ["storage_core.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/policy/internalversion:go_default_library",
        "//pkg/kubelet/client:go_default_library",
        "//pkg/master/ports:go_default_library",
        "//pkg/registry/core/componentstatus:go_default_library",
        "//pkg/registry/core/configmap/storage:go_default_library",
        "//pkg/registry/core/endpoint:go_default_library",
        "//pkg/registry/core/endpoint/storage:go_default_library",
        "//pkg/registry/core/event/storage:go_default_library",
        "//pkg/registry/core/limitrange/storage:go_default_library",
        "//pkg/registry/core/namespace/storage:go_default_library",
        "//pkg/registry/core/node/storage:go_default_library",
        "//pkg/registry/core/persistentvolume/storage:go_default_library",
        "//pkg/registry/core/persistentvolumeclaim/storage:go_default_library",
        "//pkg/registry/core/pod/storage:go_default_library",
        "//pkg/registry/core/podtemplate/storage:go_default_library",
        "//pkg/registry/core/rangeallocation:go_default_library",
        "//pkg/registry/core/replicationcontroller/storage:go_default_library",
        "//pkg/registry/core/resourcequota/storage:go_default_library",
        "//pkg/registry/core/secret/storage:go_default_library",
        "//pkg/registry/core/service:go_default_library",
        "//pkg/registry/core/service/allocator:go_default_library",
        "//pkg/registry/core/service/allocator/storage:go_default_library",
        "//pkg/registry/core/service/ipallocator:go_default_library",
        "//pkg/registry/core/service/portallocator:go_default_library",
        "//pkg/registry/core/service/storage:go_default_library",
        "//pkg/registry/core/serviceaccount/storage:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/util/net",
        "//vendor:k8s.io/apiserver/pkg/registry/generic",
        "//vendor:k8s.io/apiserver/pkg/registry/rest",
        "//vendor:k8s.io/apiserver/pkg/server",
        "//vendor:k8s.io/apiserver/pkg/storage/etcd/util",
        "//vendor:k8s.io/client-go/rest",
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
