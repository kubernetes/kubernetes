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
        "zz_generated.componentstatus.go",
        "zz_generated.configmap.go",
        "zz_generated.endpoints.go",
        "zz_generated.event.go",
        "zz_generated.limitrange.go",
        "zz_generated.namespace.go",
        "zz_generated.node.go",
        "zz_generated.persistentvolume.go",
        "zz_generated.persistentvolumeclaim.go",
        "zz_generated.pod.go",
        "zz_generated.podtemplate.go",
        "zz_generated.replicationcontroller.go",
        "zz_generated.resourcequota.go",
        "zz_generated.secret.go",
        "zz_generated.service.go",
        "zz_generated.serviceaccount.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//pkg/api/errors:go_default_library",
        "//pkg/api/v1:go_default_library",
        "//pkg/client/cache:go_default_library",
        "//pkg/labels:go_default_library",
    ],
)
