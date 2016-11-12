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
        "core_client.go",
        "doc.go",
        "endpoints.go",
        "event.go",
        "event_expansion.go",
        "generated_expansion.go",
        "limitrange.go",
        "namespace.go",
        "namespace_expansion.go",
        "node.go",
        "node_expansion.go",
        "persistentvolume.go",
        "persistentvolumeclaim.go",
        "pod.go",
        "pod_expansion.go",
        "podtemplate.go",
        "replicationcontroller.go",
        "resourcequota.go",
        "secret.go",
        "service.go",
        "service_expansion.go",
        "serviceaccount.go",
    ],
    tags = ["automanaged"],
)
