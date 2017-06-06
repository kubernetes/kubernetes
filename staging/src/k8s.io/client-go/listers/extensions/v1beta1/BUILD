package(default_visibility = ["//visibility:public"])

licenses(["notice"])

load(
    "@io_bazel_rules_go//go:def.bzl",
    "go_library",
)

go_library(
    name = "go_default_library",
    srcs = [
        "daemonset.go",
        "daemonset_expansion.go",
        "deployment.go",
        "deployment_expansion.go",
        "expansion_generated.go",
        "ingress.go",
        "podsecuritypolicy.go",
        "replicaset.go",
        "replicaset_expansion.go",
        "scale.go",
        "thirdpartyresource.go",
    ],
    tags = ["automanaged"],
    deps = [
        "//vendor/k8s.io/apimachinery/pkg/api/errors:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/apis/meta/v1:go_default_library",
        "//vendor/k8s.io/apimachinery/pkg/labels:go_default_library",
        "//vendor/k8s.io/client-go/pkg/api/v1:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/apps/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/pkg/apis/extensions/v1beta1:go_default_library",
        "//vendor/k8s.io/client-go/tools/cache:go_default_library",
    ],
)
