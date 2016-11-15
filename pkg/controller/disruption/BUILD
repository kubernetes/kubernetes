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
    srcs = ["disruption.go"],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/policy:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/core/internalversion:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/policy/internalversion:go_default_library",
        "//pkg/client/record:go_default_library",
        "//pkg/controller:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/types:go_default_library",
        "//pkg/util/intstr:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//pkg/util/workqueue:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/golang/glog",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["disruption_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/unversioned:go_default_library",
        "//pkg/apimachinery/registered:go_default_library",
        "//pkg/apis/extensions:go_default_library",
        "//pkg/apis/policy:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/record:go_default_library",
        "//pkg/controller:go_default_library",
        "//pkg/util/intstr:go_default_library",
        "//pkg/util/uuid:go_default_library",
        "//pkg/util/workqueue:go_default_library",
    ],
)
