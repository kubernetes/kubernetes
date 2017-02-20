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
        "deployment.go",
        "deployment_expansion.go",
        "doc.go",
        "extensions_client.go",
        "generated_expansion.go",
        "ingress.go",
        "podsecuritypolicy.go",
        "replicaset.go",
        "scale.go",
        "scale_expansion.go",
        "thirdpartyresource.go",
    ],
    tags = ["automanaged"],
    deps = ["//vendor:k8s.io/apimachinery/pkg/apis/meta/v1"],
)

filegroup(
    name = "package-srcs",
    srcs = glob(["**"]),
    tags = ["automanaged"],
    visibility = ["//visibility:private"],
)

filegroup(
    name = "all-srcs",
    srcs = [
        ":package-srcs",
        "//pkg/client/clientset_generated/clientset/typed/extensions/v1beta1/fake:all-srcs",
    ],
    tags = ["automanaged"],
)
