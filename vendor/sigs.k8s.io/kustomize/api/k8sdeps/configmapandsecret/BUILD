load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "configmapfactory.go",
        "factory.go",
        "secretfactory.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/api/k8sdeps/configmapandsecret",
    importpath = "sigs.k8s.io/kustomize/api/k8sdeps/configmapandsecret",
    visibility = ["//visibility:public"],
    deps = [
        "//staging/src/k8s.io/api/core/v1:go_default_library",
        "//vendor/github.com/pkg/errors:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/ifc:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/types:go_default_library",
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
    visibility = ["//visibility:public"],
)
