package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "componentstatus.go",
        "configmap.go",
        "endpoints.go",
        "event.go",
        "expansion_generated.go",
        "limitrange.go",
        "namespace.go",
        "node.go",
        "node_expansion.go",
        "persistentvolume.go",
        "persistentvolumeclaim.go",
        "pod.go",
        "podtemplate.go",
        "replicationcontroller.go",
        "replicationcontroller_expansion.go",
        "resourcequota.go",
        "secret.go",
        "service.go",
        "service_expansion.go",
        "serviceaccount.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/api:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
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
