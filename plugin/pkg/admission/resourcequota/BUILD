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
        "admission.go",
        "controller.go",
        "doc.go",
        "resource_access.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/admission:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/meta:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/internalclientset:go_default_library",
        "//pkg/quota:go_default_library",
        "//pkg/quota/install:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/storage/etcd:go_default_library",
        "//pkg/util/runtime:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//pkg/util/wait:go_default_library",
        "//pkg/util/workqueue:go_default_library",
        "//pkg/util/workqueue/prometheus:go_default_library",
        "//pkg/watch:go_default_library",
        "//vendor:github.com/golang/glog",
        "//vendor:github.com/hashicorp/golang-lru",
    ],
)

go_test(
    name = "go_default_test",
    srcs = ["admission_test.go"],
    library = "go_default_library",
    tags = ["automanaged"],
    deps = [
        "//pkg/admission:go_default_library",
        "//pkg/api:go_default_library",
        "//pkg/api/resource:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/fake:go_default_library",
        "//pkg/client/testing/core:go_default_library",
        "//pkg/quota:go_default_library",
        "//pkg/quota/evaluator/core:go_default_library",
        "//pkg/quota/generic:go_default_library",
        "//pkg/quota/install:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/runtime/schema:go_default_library",
        "//pkg/util/sets:go_default_library",
        "//vendor:github.com/hashicorp/golang-lru",
    ],
)
