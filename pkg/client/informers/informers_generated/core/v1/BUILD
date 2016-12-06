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
        "componentstatus.go",
        "configmap.go",
        "endpoints.go",
        "event.go",
        "interface.go",
        "limitrange.go",
        "namespace.go",
        "node.go",
        "persistentvolume.go",
        "persistentvolumeclaim.go",
        "pod.go",
        "podtemplate.go",
        "replicationcontroller.go",
        "resourcequota.go",
        "secret.go",
        "service.go",
        "serviceaccount.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api/v1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/client/clientset_generated/release_1_5:go_default_library",
        "//pkg/client/informers/informers_generated/internalinterfaces:go_default_library",
        "//pkg/client/listers/core/v1:go_default_library",
        "//pkg/runtime:go_default_library",
        "//pkg/watch:go_default_library",
    ],
)
