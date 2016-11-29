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
    srcs = [
        "storage_extensions.go",
        "thirdparty_controller.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/rest:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/extensions/v1beta1:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion:go_default_library",
        "//pkg/genericapiserver:go_default_library",
        "//pkg/registry:go_default_library",
        "//pkg/registry/autoscaling/horizontalpodautoscaler/etcd:go_default_library",
        "//pkg/registry/batch/job/etcd:go_default_library",
        "//pkg/registry/extensions/controller/etcd:go_default_library",
        "//pkg/registry/extensions/daemonset/etcd:go_default_library",
        "//pkg/registry/extensions/deployment/etcd:go_default_library",
        "//pkg/registry/extensions/ingress/etcd:go_default_library",
        "//pkg/registry/extensions/networkpolicy/etcd:go_default_library",
        "//pkg/registry/extensions/podsecuritypolicy/etcd:go_default_library",
        "//pkg/registry/extensions/replicaset/etcd:go_default_library",
        "//pkg/registry/extensions/thirdpartyresource/etcd:go_default_library",
        "//pkg/registry/extensions/thirdpartyresourcedata:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["thirdparty_controller_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/registry/extensions/thirdpartyresourcedata:go_default_library",
        "//pkg/util/sets:go_default_library",
    ],
)
