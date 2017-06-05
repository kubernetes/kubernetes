package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "fake_daemonset.go",
        "fake_deployment.go",
        "fake_deployment_expansion.go",
        "fake_extensions_client.go",
        "fake_ingress.go",
        "fake_networkpolicy.go",
        "fake_podsecuritypolicy.go",
        "fake_replicaset.go",
        "fake_scale.go",
        "fake_scale_expansion.go",
        "fake_thirdpartyresource.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//pkg/apis/extensions:go_default_library",
        "//pkg/client/clientset_generated/internalclientset/typed/extensions/internalversion:go_default_library",
        "//vendor:k8s.io/apimachinery/pkg/apis/meta/v1",
        "//vendor:k8s.io/apimachinery/pkg/labels",
        "//vendor:k8s.io/apimachinery/pkg/runtime/schema",
        "//vendor:k8s.io/apimachinery/pkg/types",
        "//vendor:k8s.io/apimachinery/pkg/watch",
        "//vendor:k8s.io/client-go/rest",
        "//vendor:k8s.io/client-go/testing",
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
