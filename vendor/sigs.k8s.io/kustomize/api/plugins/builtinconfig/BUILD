load("@io_bazel_rules_go//go:def.bzl", "go_library")

go_library(
    name = "go_default_library",
    srcs = [
        "doc.go",
        "loaddefaultconfig.go",
        "namebackreferences.go",
        "transformerconfig.go",
    ],
    importmap = "k8s.io/kubernetes/vendor/sigs.k8s.io/kustomize/api/plugins/builtinconfig",
    importpath = "sigs.k8s.io/kustomize/api/plugins/builtinconfig",
    visibility = ["//visibility:public"],
    deps = [
        "//vendor/sigs.k8s.io/kustomize/api/ifc:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/plugins/builtinconfig/consts:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/resid:go_default_library",
        "//vendor/sigs.k8s.io/kustomize/api/types:go_default_library",
        "//vendor/sigs.k8s.io/yaml:go_default_library",
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
    srcs = [
        ":package-srcs",
        "//vendor/sigs.k8s.io/kustomize/api/plugins/builtinconfig/consts:all-srcs",
    ],
    tags = ["automanaged"],
    visibility = ["//visibility:public"],
)
